/*
 * Copyright 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camera2raw;

import static org.tensorflow.lite.support.common.FileUtil.*;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.app.Fragment;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.Message;
import android.os.SystemClock;
import android.support.v13.app.FragmentCompat;
import android.support.v4.app.ActivityCompat;
import android.text.method.KeyListener;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import junit.framework.Assert;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.task.core.BaseOptions;

public class Camera2RawFragment extends Fragment
        implements View.OnClickListener, FragmentCompat.OnRequestPermissionsResultCallback {

    /**
     * Conversion from screen rotation to JPEG orientation.
     */
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 0);
        ORIENTATIONS.append(Surface.ROTATION_90, 90);
        ORIENTATIONS.append(Surface.ROTATION_180, 180);
        ORIENTATIONS.append(Surface.ROTATION_270, 270);
    }

    /**
     * Request code for camera permissions.
     */
    private static final int REQUEST_CAMERA_PERMISSIONS = 1;

    /**
     * Permissions required to take a picture.
     */
    private static final String[] CAMERA_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
    };

    /**
     * Timeout for the pre-capture sequence.
     */
    private static final long PRECAPTURE_TIMEOUT_MS = 1000;

    /**
     * Tolerance when comparing aspect ratios.
     */
    private static final double ASPECT_RATIO_TOLERANCE = 0.005;

    /**
     * Max preview width that is guaranteed by Camera2 API
     */
    private static final int MAX_PREVIEW_WIDTH = 1920;

    /**
     * Max preview height that is guaranteed by Camera2 API
     */
    private static final int MAX_PREVIEW_HEIGHT = 1080;

    /**
     * Tag for the {@link Log}.
     */
    private static final String TAG = "Camera2RawFragment";

    /**
     * Camera state: Device is closed.
     */
    private static final int STATE_CLOSED = 0;

    /**
     * Camera state: Device is opened, but is not capturing.
     */
    private static final int STATE_OPENED = 1;

    /**
     * Camera state: Showing camera preview.
     */
    private static final int STATE_PREVIEW = 2;

    /**
     * Camera state: Waiting for 3A convergence before capturing a photo.
     */
    private static final int STATE_WAITING_FOR_3A_CONVERGENCE = 3;

    /**
     * An {@link OrientationEventListener} used to determine when device rotation has occurred.
     * This is mainly necessary for when the device is rotated by 180 degrees, in which case
     * onCreate or onConfigurationChanged is not called as the view dimensions remain the same,
     * but the orientation of the has changed, and thus the preview rotation must be updated.
     */
    private OrientationEventListener mOrientationListener;


    private Handler handler = new Handler(Looper.getMainLooper());

    /**
     * {@link TextureView.SurfaceTextureListener} handles several lifecycle events of a
     * {@link TextureView}.
     */
    private final TextureView.SurfaceTextureListener mSurfaceTextureListener
            = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
            synchronized (mCameraStateLock) {
                mPreviewSize = null;
            }
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture texture) {
        }

    };

    /**
     * An {@link AutoFitTextureView} for camera preview.
     */
    private AutoFitTextureView mTextureView;

    /**
     * An additional thread for running tasks that shouldn't block the UI.  This is used for all
     * callbacks from the {@link CameraDevice} and {@link CameraCaptureSession}s.
     */
    private HandlerThread mBackgroundThread;

    /**
     * A counter for tracking corresponding {@link CaptureRequest}s and {@link CaptureResult}s
     * across the {@link CameraCaptureSession} capture callbacks.
     */
    private final AtomicInteger mRequestCounter = new AtomicInteger();

    /**
     * A {@link Semaphore} to prevent the app from exiting before closing the camera.
     */
    private final Semaphore mCameraOpenCloseLock = new Semaphore(1);

    /**
     * A lock protecting camera state.
     */
    private final Object mCameraStateLock = new Object();

    // *********************************************************************************************
    // State protected by mCameraStateLock.
    //
    // The following state is used across both the UI and background threads.  Methods with "Locked"
    // in the name expect mCameraStateLock to be held while calling.

    /**
     * ID of the current {@link CameraDevice}.
     */
    private String mCameraId;

    /**
     * A {@link CameraCaptureSession } for camera preview.
     */
    private CameraCaptureSession mCaptureSession;

    /**
     * A reference to the open {@link CameraDevice}.
     */
    private CameraDevice mCameraDevice;

    /**
     * The {@link Size} of camera preview.
     */
    private Size mPreviewSize;

    /**
     * The {@link CameraCharacteristics} for the currently configured camera device.
     */
    private CameraCharacteristics mCharacteristics;

    /**
     * A {@link Handler} for running tasks in the background.
     */
    private Handler mBackgroundHandler;

    /**
     * A reference counted holder wrapping the {@link ImageReader} that handles JPEG image
     * captures. This is used to allow us to clean up the {@link ImageReader} when all background
     * tasks using its {@link Image}s have completed.
     */

    /**
     * A reference counted holder wrapping the {@link ImageReader} that handles RAW image captures.
     * This is used to allow us to clean up the {@link ImageReader} when all background tasks using
     * its {@link Image}s have completed.
     */
    private RefCountedAutoCloseable<ImageReader> mRawImageReader;


    /**
     * {@link CaptureRequest.Builder} for the camera preview
     */
    private CaptureRequest.Builder mPreviewRequestBuilder;

    /**
     * The state of the camera device.
     *
     * @see #
     */
    private int mState = STATE_CLOSED;

    /**
     * Timer to use with pre-capture sequence to ensure a timely capture if 3A convergence is
     * taking too long.
     */
    private long mCaptureTimer;

    //**********************************************************************************************

    private Image mImage;
    byte[] imageBytes = new byte[3000*4000*2];
    float[] outputArray = new float[3000*4000]; //1500,2000,4
    byte[] outputArray2 = new byte[750*1000*4]; //1500,2000,4
    float[] inputTensor = new float[1*3*750*1000];
    float[] outputTensor = new float[1*3*750*1000];

    int pend=0;
    List<ByteBuffer> listBuffer = new ArrayList<>();

    Bitmap randomBitmap = Bitmap.createBitmap(1000,750,Bitmap.Config.ARGB_8888);
    CaptureRequest.Builder captureBuilder;
    CaptureRequest mCaptureRequest;
    ImageView mImageView;

    SeekBar mSeekBarShutterSpeed;
    SeekBar mSeekBarISO;
    TextView mTextViewShutter;
    TextView mTextViewISO;
    long maxShutterSpeed;

    //TFlite
    TensorImage input = new TensorImage(DataType.FLOAT32);
//    TensorImage output = new TensorImage(DataType.FLOAT32);
    TensorBuffer probabilityBuffer =
            TensorBuffer.createFixedSize(new int[]{1, 750,1000,3}, DataType.FLOAT32);
    Interpreter tflite;// = new Interpreter(tfliteModel);
    int[] inpShape = {1,750,1000,3};
    //**********************************************************************************************


    /**
     * {@link CameraDevice.StateCallback} is called when the currently active {@link CameraDevice}
     * changes its state.
     */
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice cameraDevice) {
            // This method is called when the camera is opened.  We start camera preview here if
            // the TextureView displaying this has been set up.
            synchronized (mCameraStateLock) {
                mState = STATE_OPENED;
                mCameraOpenCloseLock.release();
                mCameraDevice = cameraDevice;

            }
        }

        @Override
        public void onDisconnected(CameraDevice cameraDevice) {
            synchronized (mCameraStateLock) {
                mState = STATE_CLOSED;
                mCameraOpenCloseLock.release();
                cameraDevice.close();
                mCameraDevice = null;
            }
        }

        @Override
        public void onError(CameraDevice cameraDevice, int error) {
            Log.e(TAG, "Received camera device error: " + error);
            synchronized (mCameraStateLock) {
                mState = STATE_CLOSED;
                mCameraOpenCloseLock.release();
                cameraDevice.close();
                mCameraDevice = null;
            }
            Activity activity = getActivity();
            if (null != activity) {
                activity.finish();
            }
        }

    };

    /**
     * This a callback object for the {@link ImageReader}. "onImageAvailable" will be called when a
     * JPEG image is ready to be saved.
     */


    /**
     * This a callback object for the {@link ImageReader}. "onImageAvailable" will be called when a
     * RAW image is ready to be saved.
     */
    private final ImageReader.OnImageAvailableListener mOnRawImageAvailableListener
            = new ImageReader.OnImageAvailableListener() {

        @Override
        public void onImageAvailable(ImageReader reader) {
            pend = pend+1;
            Log.e("Filming", "onImageAva");
            mImage = mRawImageReader.get().acquireLatestImage();
            ByteBuffer buffer = mImage.getPlanes()[0].getBuffer();

            buffer.get(imageBytes);
//            listBuffer.add(buffer);
            mImage.close();
        }

    };



    /**
     * A {@link CameraCaptureSession.CaptureCallback} that handles the still JPEG and RAW capture
     * request.
     */

    public void setSeekBarISO() {
        long maxISO = mCharacteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE).getUpper();
        long minISO = mCharacteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE).getLower();
        final int isoStep = 20;
        mSeekBarISO.setMax((int)(maxISO/isoStep));
        mSeekBarISO.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int iso = progress*isoStep;
                Log.e("error", "setISO"+iso);
                mTextViewISO.setX(seekBar.getThumb().getBounds().left);
                mTextViewISO.setText(String.valueOf(iso));
                captureBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, iso);
                mCaptureRequest = captureBuilder.build();
                try {
                    mCaptureSession.setRepeatingRequest(mCaptureRequest, mCaptureCallback, mBackgroundHandler);
                } catch (CameraAccessException e) {
                    e.printStackTrace();
                }

            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                //write custom code to on start progress
            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });
    }


    public void setSeekBarShutterSpeed(){
        long maxShutterSpeed = mCharacteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE).getLower();
        int ranShutterSpeedMax = (int) (Math.log((double)maxShutterSpeed) / Math.log(2));//15;
        int ranShutterSpeedMin = 7;
        int arrayLen = (ranShutterSpeedMax-ranShutterSpeedMin);

        final long[] arrayShutterSpeed = new long[arrayLen];
        for(int i=0;i<arrayLen;i++){
            float ss = (float) Math.pow(2, i+ranShutterSpeedMin);
            arrayShutterSpeed[i]= (long)(1/ss*1000000000);
            Log.e("error", "shutterRan"+i+" "+ss+" "+arrayShutterSpeed[i]);
        }


        mSeekBarShutterSpeed.setMax(ranShutterSpeedMax-ranShutterSpeedMin-1);
        mSeekBarShutterSpeed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                long shutterSpeed = arrayShutterSpeed[progress];
                int s2 = (int) (1/(((float)shutterSpeed)/1000000000));

                mTextViewShutter.setX(seekBar.getThumb().getBounds().left);
                mTextViewShutter.setText("1/"+s2);

                Log.e("error", "seekbarx"+progress+"ss:"+s2);

                captureBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterSpeed);
                mCaptureRequest = captureBuilder.build();
                try {
                    mCaptureSession.setRepeatingRequest(mCaptureRequest, mCaptureCallback, mBackgroundHandler);
                } catch (CameraAccessException e) {
                    e.printStackTrace();
                }

            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                //write custom code to on start progress
            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });




    }

    public void rawToVisualBitmap(){
        for (int i=0; i<imageBytes.length/2; i++){
            float out =  (float) ((imageBytes[i*2] & 0xFF) | ((imageBytes[i*2+1] & 0xFF) << 8));
            out = (out-64)/(1023-64)*255;
            outputArray[i]=((out));
        }
        for (int i=0; i< 3000; i=i+4){
            for (int j=0; j<4000; j=j+4) {
                float g1 = (float) (outputArray[i * 4000 + j]);
                float r = (float) (outputArray[(i+1) * 4000 + j]);
                float b = (float) (outputArray[i * 4000 + j+1]);
                float g2 = (float) (outputArray[(i+1) * 4000 + j+1]);
                float g = (float) ((g1+g2)/2);
//                inputTensor[(i/4)*3000+(3*j/4)+0] = r;
//                inputTensor[(i/4)*3000+(3*j/4)+1] = g;
//                inputTensor[(i/4)*3000+(3*j/4)+2] = b;
//                inputTensor[(i/4)*1000*4+(j/4)*4+3] = g;
                inputTensor[((0*750*1000)+(i/4)*1000+(j/4))] = r;
                inputTensor[((1*750*1000)+(i/4)*1000+(j/4))] = g;
                inputTensor[((2*750*1000)+(i/4)*1000+(j/4))] = b;

            }
        }
//        for (int i=0; i< 3000; i=i+4){
//            for (int j=0; j<4000; j=j+4) {
//                float g1 = (float) (outputArray[i * 4000 + j]);
//                byte b = (byte) (outputArray[(i+1) * 4000 + j]);
//                byte r = (byte) (outputArray[i * 4000 + j+1]);
//                float g2 = (float) (outputArray[(i+1) * 4000 + j+1]);
//                byte g = (byte) ((g1+g2)/2);
//                outputArray2[(i/4)*1000*4+(j/4)*4+0] = r;
//                outputArray2[(i/4)*1000*4+(j/4)*4+1] = g;
//                outputArray2[(i/4)*1000*4+(j/4)*4+2] = b;
//                outputArray2[(i/4)*1000*4+(j/4)*4+3] = -1;
//
//            }
//        }

//        randomBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(outputArray2));
//        mImageView.setImageBitmap(randomBitmap);

    }
    private final CameraCaptureSession.CaptureCallback mCaptureCallback
            = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureStarted(CameraCaptureSession session, CaptureRequest request,
                                     long timestamp, long frameNumber) {

            // Look up the ImageSaverBuilder for this request and update it with the file name
            // based on the capture start time.
            }

        @Override
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request,
                                       TotalCaptureResult result) {
            Log.e("error", "exp:"+result.get(CaptureResult.SENSOR_SENSITIVITY));
//            float[][][][] input = new float[1][4][750][1500];
//            for(int c=0; c<4;c++){
//                for(int h=0;h<750;h++){
//                    for(int w=0;w<1500;w++){
//                        input[0][c][h][w]= 1;//outputArray2[c*750*1500+h*1500+w];
//                    }
//                }
//            }
//            if(pend==30) {
//                pend = 0;
//                byte[] x = new byte[3000*4000*2];
//                Log.e("error", "aa");
//                for (final ByteBuffer bb : listBuffer) {
////                    try {
////                        Thread.sleep(2000);
////                    } catch (InterruptedException e) {
////                        e.printStackTrace();
////                    }
//                    bb.get(x);
//                    Log.e("error", "pp");
//                    for (int i = 0; i < x.length / 2; i++) {
//                        float out = (float) ((x[i * 2] & 0xFF) | ((x[i * 2 + 1] & 0xFF) << 8));
//                        out = (out - 64) / (1023 - 64) * 255;
//                        outputArray[i] = ((out));
//                    }
//
//                    for (int i = 0; i < 3000; i = i + 4) {
//                        for (int j = 0; j < 4000; j = j + 4) {
//                            float g1 = (float) (outputArray[i * 4000 + j]);
//                            byte b = (byte) (outputArray[(i + 1) * 4000 + j]);
//                            byte r = (byte) (outputArray[i * 4000 + j + 1]);
//                            float g2 = (float) (outputArray[(i + 1) * 4000 + j + 1]);
//                            byte g = (byte) ((g1 + g2) / 2);
//                            outputArray2[(i / 4) * 1000 * 4 + (j / 4) * 4 + 0] = r;
//                            outputArray2[(i / 4) * 1000 * 4 + (j / 4) * 4 + 1] = g;
//                            outputArray2[(i / 4) * 1000 * 4 + (j / 4) * 4 + 2] = b;
//                            outputArray2[(i / 4) * 1000 * 4 + (j / 4) * 4 + 3] = -1;
//
//                        }
//                    }
//                    Log.e("error", "pix:"+outputArray2[0]);
//                    randomBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(outputArray2));
//                    mImageView.setImageBitmap(randomBitmap);
//                    //what ever you do here will be done after 3 seconds delay.
//
//
//                }
//            }
            rawToVisualBitmap();
            input.load(inputTensor, inpShape);
//            try {
//                int[] inpShape = {1,750,1000,3};
//                TensorImage input = new TensorImage(DataType.FLOAT32);
//                input.load(inputTensor, inpShape);
//
//                TensorImage output = new TensorImage(DataType.FLOAT32);
//                output.load(outputTensor, inpShape);
//
//                TensorBuffer probabilityBuffer =
//                        TensorBuffer.createFixedSize(new int[]{1, 750,1000,3}, DataType.FLOAT32);
//
//                MappedByteBuffer tfliteModel;
//                tfliteModel = loadModelFile();
//                Interpreter tflite = new Interpreter(tfliteModel);
//                Log.e("error", "load pmrid model");
//
//                Log.e("error", "init input output");

                tflite.run(input.getBuffer(), probabilityBuffer.getBuffer());
                Log.e("error", "finished run");

                outputTensor = probabilityBuffer.getFloatArray();
                for (int i=0; i< 750; i++){
                    for (int j=0; j<1000; j++) {
//                        byte r = (byte) (outputTensor[i * 3000 + (3*j)]);
//                        byte g = (byte) (outputTensor[i * 3000 + (3*j)+1]);
//                        byte b = (byte) (outputTensor[i * 3000 + (3*j)+2]);
                        byte r = (byte) (outputTensor[(0*750*1000)+i*1000 + j]);
                        byte g = (byte) (outputTensor[(1*750*1000)+i*1000 + j]);
                        byte b = (byte) (outputTensor[(2*750*1000)+i*1000 + j]);
                        outputArray2[i*4000+(4*j)+0] = r;
                        outputArray2[i*4000+(4*j)+1] = g;
                        outputArray2[i*4000+(4*j)+2] = b;
                        outputArray2[i*4000+(4*j)+3] = -1;

                    }
                }
//                randomBitmap = output.getBitmap();
                randomBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(outputArray2));
                mImageView.setImageBitmap(randomBitmap);
//                Log.e("error", "outshape"+(outTensor.shape()[2])+(outTensor.shape()[3]));
//            Log.e("error", String.valueOf(tflite.getOutputIndex((String)"output")));
//            } catch (IOException e) {
//                e.printStackTrace();
//            }

        }

        @Override
        public void onCaptureFailed(CameraCaptureSession session, CaptureRequest request,
                                    CaptureFailure failure) {
            int requestId = (int) request.getTag();
            synchronized (mCameraStateLock) {
            }
            showToast("Capture failed!");
        }

    };

    /**
     * A {@link Handler} for showing {@link Toast}s on the UI thread.
     */
    private final Handler mMessageHandler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(Message msg) {
            Activity activity = getActivity();
            if (activity != null) {
                Toast.makeText(activity, (String) msg.obj, Toast.LENGTH_SHORT).show();
            }
        }
    };

    public static Camera2RawFragment newInstance() {
        return new Camera2RawFragment();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {


        final View v = inflater.inflate(R.layout.fragment_camera2_basic, container, false);
        v.setOnKeyListener(new View.OnKeyListener() {
            @Override
            public boolean onKey(View v, int keyCode, KeyEvent event) {
                // KeyEvent.ACTION_DOWN以外のイベントを無視する
                // （これがないとKeyEvent.ACTION_UPもフックしてしまう）
                if(event.getAction() != KeyEvent.ACTION_DOWN) {
                    return false;
                }

                switch(keyCode) {
                    case KeyEvent.KEYCODE_VOLUME_UP:
                        // TODO:音量増加キーが押された時のイベント
                        Log.e("error", "press");
                        return true;
                    case KeyEvent.KEYCODE_VOLUME_DOWN:
                        // TODO:音量減少キーが押された時のイベント
                        return true;
                    default:
                        return false;
                }
            }
        });

        // View#setFocusableInTouchModeでtrueをセットしておくこと
        v.setFocusableInTouchMode(true);
        return v;
    }
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor= getResources().getAssets().openFd("pmrid.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }
    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        Log.e("error", "onViewCreated2");
        List<byte[]> array = new ArrayList<>();
        for(int i=0;i<5;i++){
            array.add(new byte[3000*4000*2]);

        }


        try {
            int numThreads = 4;
            Interpreter.Options tfLiteOptions = new Interpreter.Options();
            tfLiteOptions.setNumThreads(numThreads);
            MappedByteBuffer tfliteModel;
            tfliteModel = loadModelFile();
            tflite = new Interpreter(tfliteModel, tfLiteOptions);
        } catch (IOException e) {
            e.printStackTrace();
        }


        view.findViewById(R.id.picture).setOnClickListener(this);
//        view.findViewById(R.id.info).setOnClickListener(this);
        mTextureView = (AutoFitTextureView) view.findViewById(R.id.texture);
        mImageView = (ImageView) view.findViewById(R.id.imageView);
        mSeekBarShutterSpeed = view.findViewById(R.id.seekBarShutterSpeed);
        mSeekBarISO = view.findViewById(R.id.seekBarISO);
        mTextViewShutter = view.findViewById(R.id.textViewShutterSpeed);
        mTextViewISO = view.findViewById(R.id.textViewISO);
        // Setup a new OrientationEventListener.  This is used to handle rotation events like a
        // 180 degree rotation that do not normally trigger a call to onCreate to do view re-layout
        // or otherwise cause the preview TextureView's size to change.
        mOrientationListener = new OrientationEventListener(getActivity(),
                SensorManager.SENSOR_DELAY_NORMAL) {
            @Override
            public void onOrientationChanged(int orientation) {
                if (mTextureView != null && mTextureView.isAvailable()) {
                    configureTransform(mTextureView.getWidth(), mTextureView.getHeight());
                }
            }
        };
        if (mTextureView.isAvailable()) {
            configureTransform(mTextureView.getWidth(), mTextureView.getHeight());
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    @Override
    public void onResume() {
        super.onResume();
//        startBackgroundThread();
        openCamera();

        Log.e("error", "onResume");

    }

    @Override
    public void onPause() {
        if (mOrientationListener != null) {
            mOrientationListener.disable();
        }
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSIONS) {
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    showMissingPermissionError();
                    return;
                }
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.picture: {
                captureStillPictureLocked();
//                setSeekBarShutterSpeed(view);
                break;
            }
        }
    }

    /**
     * Sets up state related to camera that is needed before opening a {@link CameraDevice}.
     */
    private boolean setUpCameraOutputs() {
        Log.e("error", "setUpCameraOutputs");
        Activity activity = getActivity();
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        if (manager == null) {
            ErrorDialog.buildErrorDialog("This device doesn't support Camera2 API.").
                    show(getFragmentManager(), "dialog");
            return false;
        }
        try {
            // Find a CameraDevice that supports RAW captures, and configure state.
            for (String cameraId : manager.getCameraIdList()) {
                CameraCharacteristics characteristics
                        = manager.getCameraCharacteristics(cameraId);

                // We only use a camera that supports RAW in this sample.
                if (!contains(characteristics.get(
                                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES),
                        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)) {
                    continue;
                }

                StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                // For still image captures, we use the largest available size.


                Size largestRaw = Collections.max(
                        Arrays.asList(map.getOutputSizes(ImageFormat.RAW_SENSOR)),
                        new CompareSizesByArea());

                synchronized (mCameraStateLock) {
                    // Set up ImageReaders for JPEG and RAW outputs.  Place these in a reference
                    // counted wrapper to ensure they are only closed when all background tasks
                    // using them are finished.

                    if (mRawImageReader == null || mRawImageReader.getAndRetain() == null) {
                        mRawImageReader = new RefCountedAutoCloseable<>(
                                ImageReader.newInstance(largestRaw.getWidth(),
                                        largestRaw.getHeight(), ImageFormat.RAW_SENSOR, /*maxImages*/ 1));
                    }
                    mRawImageReader.get().setOnImageAvailableListener(
                            mOnRawImageAvailableListener, mBackgroundHandler);

                    mCharacteristics = characteristics;
                    mCameraId = cameraId;
                }
                return true;
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }

        // If we found no suitable cameras for capturing RAW, warn the user.
        ErrorDialog.buildErrorDialog("This device doesn't support capturing RAW photos").
                show(getFragmentManager(), "dialog");
        return false;
    }

    /**
     * Opens the camera specified by {@link #mCameraId}.
     */
    @SuppressWarnings("MissingPermission")
    private void openCamera() {
        Log.e("error","openCamera");
        if (!setUpCameraOutputs()) {
            return;
        }
        if (!hasAllPermissionsGranted()) {
            requestCameraPermissions();
            return;
        }

        Activity activity = getActivity();
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            // Wait for any previously running session to finish.

            // Attempt to open the camera. mStateCallback will be called on the background handler's
            // thread when this succeeds or fails.
            manager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }

    }

    /**
     * Requests permissions necessary to use camera and save pictures.
     */
    private void requestCameraPermissions() {
        if (shouldShowRationale()) {
            PermissionConfirmationDialog.newInstance().show(getChildFragmentManager(), "dialog");
        } else {
            FragmentCompat.requestPermissions(this, CAMERA_PERMISSIONS, REQUEST_CAMERA_PERMISSIONS);
        }
    }

    /**
     * Tells whether all the necessary permissions are granted to this app.
     *
     * @return True if all the required permissions are granted.
     */
    private boolean hasAllPermissionsGranted() {
        for (String permission : CAMERA_PERMISSIONS) {
            if (ActivityCompat.checkSelfPermission(getActivity(), permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    /**
     * Gets whether you should show UI with rationale for requesting the permissions.
     *
     * @return True if the UI should be shown.
     */
    private boolean shouldShowRationale() {
        for (String permission : CAMERA_PERMISSIONS) {
            if (FragmentCompat.shouldShowRequestPermissionRationale(this, permission)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Shows that this app really needs the permission and finishes the app.
     */
    private void showMissingPermissionError() {
        Activity activity = getActivity();
        if (activity != null) {
            Toast.makeText(activity, R.string.request_permission, Toast.LENGTH_SHORT).show();
            activity.finish();
        }
    }

    /**
     * Closes the current {@link CameraDevice}.
     */
    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            synchronized (mCameraStateLock) {

                // Reset state and clean up resources used by the camera.
                // Note: After calling this, the ImageReaders will be closed after any background
                // tasks saving Images from these readers have been completed.
                mState = STATE_CLOSED;
                if (null != mCaptureSession) {
                    mCaptureSession.close();
                    mCaptureSession = null;
                }
                if (null != mCameraDevice) {
                    mCameraDevice.close();
                    mCameraDevice = null;
                }

                if (null != mRawImageReader) {
                    mRawImageReader.close();
                    mRawImageReader = null;
                }
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    /**
     * Starts a background thread and its {@link Handler}.
     */
    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        synchronized (mCameraStateLock) {
            mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
        }
    }

    /**
     * Stops the background thread and its {@link Handler}.
     */
    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            synchronized (mCameraStateLock) {
                mBackgroundHandler = null;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Creates a new {@link CameraCaptureSession} for camera preview.
     * <p/>
     * Call this only with {@link #mCameraStateLock} held.
     */
    private void createCameraPreviewSessionLocked() {
        try {
            Log.e("error", "createCameraPreviewSessionLocked");
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            // We configure the size of default buffer to be the size of camera preview we want.
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());

            // This is the output Surface we need to start preview.
            Surface surface = new Surface(texture);

            // We set up a CaptureRequest.Builder with the output Surface.
//            mPreviewRequestBuilder
//                    = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
//            mPreviewRequestBuilder.addTarget(surface);
            // Here, we create a CameraCaptureSession for camera preview.
            mCameraDevice.createCaptureSession(Arrays.asList(surface,

                            mRawImageReader.get().getSurface()), new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                            synchronized (mCameraStateLock) {
                                // The camera is already closed
                                mCaptureSession = cameraCaptureSession;
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
                            showToast("Failed to configure camera.");
                        }
                    }, mBackgroundHandler
            );
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
//        captureStillPictureLocked();
    }

    private void configureTransform(int viewWidth, int viewHeight) {
        Log.e("error", "configureTransform");
        Activity activity = getActivity();
        synchronized (mCameraStateLock) {
            if (null == mTextureView || null == activity) {
                return;
            }

            StreamConfigurationMap map = mCharacteristics.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

            // For still image captures, we always use the largest available size.
            Size largestJpeg = Collections.max(Arrays.asList(map.getOutputSizes(ImageFormat.JPEG)),
                    new CompareSizesByArea());

            // Find the rotation of the device relative to the native device orientation.
            int deviceRotation = activity.getWindowManager().getDefaultDisplay().getRotation();
            Point displaySize = new Point();
            activity.getWindowManager().getDefaultDisplay().getSize(displaySize);

            // Find the rotation of the device relative to the camera sensor's orientation.
            int totalRotation = sensorToDeviceRotation(mCharacteristics, deviceRotation);

            // Swap the view dimensions for calculation as needed if they are rotated relative to
            // the sensor.
            boolean swappedDimensions = totalRotation == 90 || totalRotation == 270;
            int rotatedViewWidth = viewWidth;
            int rotatedViewHeight = viewHeight;
            int maxPreviewWidth = displaySize.x;
            int maxPreviewHeight = displaySize.y;

            if (swappedDimensions) {
                rotatedViewWidth = viewHeight;
                rotatedViewHeight = viewWidth;
                maxPreviewWidth = displaySize.y;
                maxPreviewHeight = displaySize.x;
            }

            // Preview should not be larger than display size and 1080p.
            if (maxPreviewWidth > MAX_PREVIEW_WIDTH) {
                maxPreviewWidth = MAX_PREVIEW_WIDTH;
            }

            if (maxPreviewHeight > MAX_PREVIEW_HEIGHT) {
                maxPreviewHeight = MAX_PREVIEW_HEIGHT;
            }

            // Find the best preview size for these view dimensions and configured JPEG size.
            Size previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class),
                    rotatedViewWidth, rotatedViewHeight, maxPreviewWidth, maxPreviewHeight,
                    largestJpeg);

            if (swappedDimensions) {
                mTextureView.setAspectRatio(
                        previewSize.getHeight(), previewSize.getWidth());
            } else {
                mTextureView.setAspectRatio(
                        previewSize.getWidth(), previewSize.getHeight());
            }

            // Find rotation of device in degrees (reverse device orientation for front-facing
            // cameras).
            int rotation = (mCharacteristics.get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_FRONT) ?
                    (360 + ORIENTATIONS.get(deviceRotation)) % 360 :
                    (360 - ORIENTATIONS.get(deviceRotation)) % 360;

            Matrix matrix = new Matrix();
            RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
            RectF bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());
            float centerX = viewRect.centerX();
            float centerY = viewRect.centerY();

            if (Surface.ROTATION_90 == deviceRotation || Surface.ROTATION_270 == deviceRotation) {
                bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
                matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
                float scale = Math.max(
                        (float) viewHeight / previewSize.getHeight(),
                        (float) viewWidth / previewSize.getWidth());
                matrix.postScale(scale, scale, centerX, centerY);

            }
            matrix.postRotate(rotation, centerX, centerY);

            mTextureView.setTransform(matrix);

            // Start or restart the active capture session if the preview was initialized or
            // if its aspect ratio changed significantly.
            if (mPreviewSize == null || !checkAspectsEqual(previewSize, mPreviewSize)) {
                mPreviewSize = previewSize;
                if (mState != STATE_CLOSED) {
                    createCameraPreviewSessionLocked();
                }
            }
        }
    }

    /**
     * Initiate a still image capture.
     * <p/>
     * This function sends a capture request that initiates a pre-capture sequence in our state
     * machine that waits for auto-focus to finish, ending in a "locked" state where the lens is no
     * longer moving, waits for auto-exposure to choose a good exposure value, and waits for
     * auto-white-balance to converge.
     */

    /**
     * Send a capture request to the camera device that initiates a capture targeting the JPEG and
     * RAW outputs.
     * <p/>
     * Call this only with {@link #mCameraStateLock} held.
     */
    private void captureStillPictureLocked() {
        Log.e("error", "captureStillPictureLocked");
//        mState = STATE_WAITING_FOR_3A_CONVERGENCE;
        try {

            // This is the CaptureRequest.Builder that we use to take a picture.
//            final CaptureRequest.Builder captureBuilder =

            captureBuilder= mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
            captureBuilder.addTarget(mRawImageReader.get().getSurface());
            captureBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
            captureBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            captureBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, 6400);
            captureBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, (long) 5600000);//(long)(0.02 *1000000000));
            // Use the same AE and AF modes as the preview.
//            setup3AControlsLocked(captureBuilder);
            // Set request tag to easily track results in callbacks.
//            captureBuilder.setTag(mRequestCounter.getAndIncrement());
//            mCaptureSession.setRepeatingRequest(captureBuilder.build(), mCaptureCallback, mBackgroundHandler);
            mCaptureRequest = captureBuilder.build();


            List<CaptureRequest> listCaptureRequest = new ArrayList<>();
            for(int i =0; i<30; i++){
                listCaptureRequest.add(mCaptureRequest);
            }
            mCaptureSession.captureBurst(listCaptureRequest, mCaptureCallback, mBackgroundHandler);
//            mCaptureSession.capture(mCaptureRequest, mCaptureCallback, mBackgroundHandler);

            setSeekBarShutterSpeed();
            setSeekBarISO();
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }



    static class CompareSizesByArea implements Comparator<Size> {

        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }

    }

    /**
     * A dialog fragment for displaying non-recoverable errors; this {@ling Activity} will be
     * finished once the dialog has been acknowledged by the user.
     */
    public static class ErrorDialog extends DialogFragment {

        private String mErrorMessage;

        public ErrorDialog() {
            mErrorMessage = "Unknown error occurred!";
        }

        // Build a dialog with a custom message (Fragments require default constructor).
        public static ErrorDialog buildErrorDialog(String errorMessage) {
            ErrorDialog dialog = new ErrorDialog();
            dialog.mErrorMessage = errorMessage;
            return dialog;
        }

        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Activity activity = getActivity();
            return new AlertDialog.Builder(activity)
                    .setMessage(mErrorMessage)
                    .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialogInterface, int i) {
                            activity.finish();
                        }
                    })
                    .create();
        }
    }

    /**
     * A wrapper for an {@link AutoCloseable} object that implements reference counting to allow
     * for resource management.
     */
    public static class RefCountedAutoCloseable<T extends AutoCloseable> implements AutoCloseable {
        private T mObject;
        private long mRefCount = 0;

        /**
         * Wrap the given object.
         *
         * @param object an object to wrap.
         */
        public RefCountedAutoCloseable(T object) {
            if (object == null) throw new NullPointerException();
            mObject = object;
        }

        /**
         * Increment the reference count and return the wrapped object.
         *
         * @return the wrapped object, or null if the object has been released.
         */
        public synchronized T getAndRetain() {
            if (mRefCount < 0) {
                return null;
            }
            mRefCount++;
            return mObject;
        }

        /**
         * Return the wrapped object.
         *
         * @return the wrapped object, or null if the object has been released.
         */
        public synchronized T get() {
            return mObject;
        }

        /**
         * Decrement the reference count and release the wrapped object if there are no other
         * users retaining this object.
         */
        @Override
        public synchronized void close() {
            if (mRefCount >= 0) {
                mRefCount--;
                if (mRefCount < 0) {
                    try {
                        mObject.close();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    } finally {
                        mObject = null;
                    }
                }
            }
        }
    }

    private static Size chooseOptimalSize(Size[] choices, int textureViewWidth,
            int textureViewHeight, int maxWidth, int maxHeight, Size aspectRatio) {
        // Collect the supported resolutions that are at least as big as the preview Surface
        List<Size> bigEnough = new ArrayList<>();
        // Collect the supported resolutions that are smaller than the preview Surface
        List<Size> notBigEnough = new ArrayList<>();
        int w = aspectRatio.getWidth();
        int h = aspectRatio.getHeight();
        for (Size option : choices) {
            if (option.getWidth() <= maxWidth && option.getHeight() <= maxHeight &&
                    option.getHeight() == option.getWidth() * h / w) {
                if (option.getWidth() >= textureViewWidth &&
                    option.getHeight() >= textureViewHeight) {
                    bigEnough.add(option);
                } else {
                    notBigEnough.add(option);
                }
            }
        }

        // Pick the smallest of those big enough. If there is no one big enough, pick the
        // largest of those not big enough.
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else if (notBigEnough.size() > 0) {
            return Collections.max(notBigEnough, new CompareSizesByArea());
        } else {
            Log.e(TAG, "Couldn't find any suitable preview size");
            return choices[0];
        }
    }


    private static boolean contains(int[] modes, int mode) {
        if (modes == null) {
            return false;
        }
        for (int i : modes) {
            if (i == mode) {
                return true;
            }
        }
        return false;
    }

    private static boolean checkAspectsEqual(Size a, Size b) {
        double aAspect = a.getWidth() / (double) a.getHeight();
        double bAspect = b.getWidth() / (double) b.getHeight();
        return Math.abs(aAspect - bAspect) <= ASPECT_RATIO_TOLERANCE;
    }


    private static int sensorToDeviceRotation(CameraCharacteristics c, int deviceOrientation) {
        int sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION);

        // Get device orientation in degrees
        deviceOrientation = ORIENTATIONS.get(deviceOrientation);

        // Reverse device orientation for front-facing cameras
        if (c.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) {
            deviceOrientation = -deviceOrientation;
        }

        // Calculate desired JPEG orientation relative to camera orientation to make
        // the image upright relative to the device orientation
        return (sensorOrientation - deviceOrientation + 360) % 360;
    }

    private void showToast(String text) {
        // We show a Toast by sending request message to mMessageHandler. This makes sure that the
        // Toast is shown on the UI thread.
        Message message = Message.obtain();
        message.obj = text;
        mMessageHandler.sendMessage(message);
    }

    public static class PermissionConfirmationDialog extends DialogFragment {

        public static PermissionConfirmationDialog newInstance() {
            return new PermissionConfirmationDialog();
        }

        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Fragment parent = getParentFragment();
            return new AlertDialog.Builder(getActivity())
                    .setMessage(R.string.request_permission)
                    .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            FragmentCompat.requestPermissions(parent, CAMERA_PERMISSIONS,
                                    REQUEST_CAMERA_PERMISSIONS);
                        }
                    })
                    .setNegativeButton(android.R.string.cancel,
                            new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    getActivity().finish();
                                }
                            })
                    .create();
        }

    }

}
