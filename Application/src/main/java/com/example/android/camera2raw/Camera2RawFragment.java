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
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
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
import android.hardware.camera2.DngCreator;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.Message;
import android.provider.ContactsContract;
import android.support.v13.app.FragmentCompat;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
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

import java.io.ByteArrayInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

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
    long toUS = 1000000000;
    int mISO;
    long mShutterSpeed;
    int mRatio=1;
    int Height = 1472;
    int Width = 1984;
    int Channel = 4;
    long gtExposure;
    int gtIso;
    Size largestRaw;
    private Image mImage;
    byte[] imageBytes;// = new byte[3000*4000*2];
//    float[] outputArray = new float[3000*4000]; //1500,2000,4
//    byte[] outputArray2 = new byte[Height*Width*4]; //1500,2000,4


    float[] inputTensor = new float[1*Channel*Height*Width];
    float[] outputTensor = new float[1*Channel*Height*Width];

    int pend=0;
    List<ByteBuffer> listBuffer = new ArrayList<>();

//    Bitmap randomBitmap = Bitmap.createBitmap(Width,Height,Bitmap.Config.ARGB_8888);
    CaptureRequest.Builder captureBuilder;
    CaptureRequest mCaptureRequest;
    ImageView mImageView;

    SeekBar mSeekBarShutterSpeed;
    SeekBar mSeekBarISO;
    TextView mTextViewShutter;
    TextView mTextViewISO;
    TextView mTextureViewAutoExp;
//    long maxShutterSpeed;

    //TFlite
    TensorBuffer input = TensorBuffer.createFixedSize(new int[] {1, Height, Width, Channel}, DataType.FLOAT32);
//    TensorImage input = new TensorImage(DataType.FLOAT32);
//    TensorImage output = new TensorImage(DataType.FLOAT32);
    TensorBuffer probabilityBuffer =
            TensorBuffer.createFixedSize(new int[]{1, Height,Width,Channel}, DataType.FLOAT32);
    Interpreter tflite;// = new Interpreter(tfliteModel);
//    int[] inpShape = {1,Height,Width,Channel};





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
//            synchronized (mCameraStateLock) {

            pend = pend + 1;
            Log.e("Filming", "onImageAva");
            mImage = mRawImageReader.get().acquireLatestImage();
            ByteBuffer buffer = mImage.getPlanes()[0].getBuffer();
            buffer.get(imageBytes);
            Log.e("error", "imageBytes:"+imageBytes[0]+" "+imageBytes[1]);
//            listBuffer.add(buffer);
            mImage.close();
            }
//        }
    };



    /**
     * A {@link CameraCaptureSession.CaptureCallback} that handles the still JPEG and RAW capture
     * request.
     */

    public void setSeekBarISO() {
        int maxISO = mCharacteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE).getUpper();
        int minISO = mCharacteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE).getLower();
        final int isoStep = 20;
        mISO = minISO;
        mSeekBarISO.setMax((int)(maxISO/isoStep));
        mSeekBarISO.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                mISO = progress*isoStep;
//                Log.e("error", "setISO"+iso);

                mRatio = (int) ((gtIso*gtExposure)/(mISO*mShutterSpeed));
                mRatio = Math.max(1, mRatio);

                mTextViewISO.setX(seekBar.getThumb().getBounds().left);
                mTextViewISO.setText(String.valueOf(mISO));
//                captureBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, iso);
//                mCaptureRequest = captureBuilder.build();
//                try {
//                    mCaptureSession.setRepeatingRequest(mCaptureRequest, mCaptureCallback, mBackgroundHandler);
//                } catch (CameraAccessException e) {
//                    e.printStackTrace();
//                }

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
        mShutterSpeed=arrayShutterSpeed[0];

        mSeekBarShutterSpeed.setMax(ranShutterSpeedMax-ranShutterSpeedMin-1);
        mSeekBarShutterSpeed.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                mShutterSpeed = arrayShutterSpeed[progress];
                int s2 = (int) (1/(((float)mShutterSpeed)/1000000000));

                mTextViewShutter.setX(seekBar.getThumb().getBounds().left);
                mTextViewShutter.setText("1/"+s2);
                mRatio = (int) ((float)(gtIso*gtExposure)/((float) (mISO*mShutterSpeed)));
                mRatio = Math.max(1, mRatio);
//                Log.e("error", "seekbarx"+progress+"ss:"+s2);
//
//                captureBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, shutterSpeed);
//                mCaptureRequest = captureBuilder.build();
//                try {
//                    mCaptureSession.setRepeatingRequest(mCaptureRequest, mCaptureCallback, mBackgroundHandler);
//                } catch (CameraAccessException e) {
//                    e.printStackTrace();
//                }

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

    public float customReLU(float input){
        if(input<0)
            input=0;
        if (input>1)
            input=1;
        return input;
    }
    public void rawToVisualBitmap(){
        Log.e("error", "raw To vi imageBytes:"+imageBytes[0]+" "+imageBytes[1]);
        for(int i=0; i<Height*2; i=i+2){
            for(int j=0; j<Width*2; j=j+2){

//                Log.e("error", "aa is:"+aa);
                float g1 =  (float) ((imageBytes[(i*2* largestRaw.getWidth())+(j*2)] & 0xFF) | ((imageBytes[(i*2*largestRaw.getWidth())+(j*2)+1] & 0xFF) << 8));
                float b =  (float) ((imageBytes[(i*2*largestRaw.getWidth())+((j+1)*2)] & 0xFF) | ((imageBytes[(i*2*largestRaw.getWidth())+((j+1)*2)+1] & 0xFF) << 8));
                float r =  (float) ((imageBytes[((i+1)*2*largestRaw.getWidth())+(j*2)] & 0xFF) | ((imageBytes[((i+1)*2*largestRaw.getWidth())+(j*2)+1] & 0xFF) << 8));
                float g2 =  (float) ((imageBytes[((i+1)*2*largestRaw.getWidth())+((j+1)*2)] & 0xFF) | ((imageBytes[((i+1)*2*largestRaw.getWidth())+((j+1)*2)+1] & 0xFF) << 8));
//                Log.e("error", "r is:"+r);
                g1 = mRatio*(g1-64)/(1024-64);
                r = mRatio*(r-64)/(1024-64);
                b = mRatio*(b-64)/(1024-64);
                g2 = mRatio*(g2-64)/(1024-64);

                g1 = (float) Math.pow(customReLU(g1), 1/2.22);
                r = (float) Math.pow(customReLU(r), 1/2.22);
                b = (float) Math.pow(customReLU(b), 1/2.22);
                g2 = (float) Math.pow(customReLU(g2),1/2.22);

//                g1 = customReLU(g1);
//                r = customReLU(r);
//                b = customReLU(b);
//                g2 = customReLU(g2);
                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 0] = r;
                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 1] = g1;
                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 2] = b;
                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 3] = g2;
            }
        }


        //uint16 to float32
//        for (int i=0; i<imageBytes.length/2; i++){
//            float out =  (float) ((imageBytes[i*2] & 0xFF) | ((imageBytes[i*2+1] & 0xFF) << 8));
//            out = ((float)mRatio)*(out-64)/(1024-64);
//            if (out<0)
//                out=0;
//            else if (out>1)
//                out=1;
//            out = (float) Math.pow(out,1/2.22);
//            outputArray[i]=((out));
//        }
//
//        //
//        for (int i=0; i< 3000; i=i+2){
//            if ((i/2) >= Height) continue;
//
//            for (int j=0; j<4000; j=j+2) {
//                if ((j/2) >= Width) continue;
//
//                float g1 = (outputArray[i * 4000 + j]);
//                float r =  (outputArray[(i+1) * 4000 + j]);
//                float b =  (outputArray[i * 4000 + j+1]);
//                float g2 = (outputArray[(i+1) * 4000 + j+1]);
//
//                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 0] = r;
//                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 1] = g1;
//                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 2] = b;
//                inputTensor[(i/2)*Width*Channel+(j/2)*Channel + 3] = g2;
//            }
//        }
    }

    private final CameraCaptureSession.CaptureCallback mPreviewCallback
            = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureStarted(CameraCaptureSession session, CaptureRequest request,
                                     long timestamp, long frameNumber) {
        }

        @Override
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request,
                                       TotalCaptureResult result) {
            gtIso = result.get(CaptureResult.SENSOR_SENSITIVITY);
            gtExposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
            long frequency = (long) (1/((float) gtExposure/toUS));
            mRatio = (int) ((float)(gtIso*gtExposure)/((float) (mISO*mShutterSpeed)));
            mRatio = Math.max(1, mRatio);
            mTextureViewAutoExp.setText("GtISO:"+gtIso+" GtFre:"+ frequency+" ratio:"+mRatio);
//            Log.e("error", "preview ISO: "+iso+" shutter:"+frequency);
        }

        @Override
        public void onCaptureFailed(CameraCaptureSession session, CaptureRequest request,
                                    CaptureFailure failure) {
        }

    };

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
            synchronized (mCameraStateLock) {

                Log.e("error", "exp:" + result.get(CaptureResult.SENSOR_SENSITIVITY));
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
                input.loadArray(inputTensor);
//            input.load(inputTensor, inpShape);
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

                ByteBuffer inpBuffer = input.getBuffer();
                ByteBuffer outBuffer = probabilityBuffer.getBuffer();
                long start = System.currentTimeMillis();
                tflite.run(inpBuffer, outBuffer);
                long runTime = System.currentTimeMillis() - start;
                Log.e("error", "Model runTime:" + runTime);
                outputTensor = probabilityBuffer.getFloatArray();
//            outputTensor=input.getFloatArray();
//               outputTensor = input.getFloatArray();

//                try {
//                    outputTensor = input.getFloatArray();
//                    byte[] saveArray = new byte[1 * Channel * Height * Width * 2];
//                    for (int i = 0; i < outputTensor.length; i++) {
//                        int tmp = (int) (outputTensor[i] * (1024 - 64) + 64);
////                        Log.e("error", "tmp is "+ outputTensor[i]);
//                        byte tmp0 = (byte) (tmp & 0xFF);
//                        byte tmp1 = (byte) ((tmp >> 8) & 0xFF);
////                    byte tmp = (byte) (outputTensor[i]*255);
//                        saveArray[(2 * i) + 0] = tmp1;
//                        saveArray[(2 * i) + 1] = tmp0;
//                    }
//                    File rawsave = new File(Environment.
//                            getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM),
//                            "TMP2" + ".dng");
//                    if (!rawsave.exists()) {
//                        rawsave.createNewFile();
//                    }
//                    FileOutputStream fos = new FileOutputStream(rawsave);
//                    fos.write(saveArray);
//                    fos.close();
//                } catch (Exception e) {
//                    Log.e(TAG, e.getMessage());
//                }

//                for (int i=0; i< Height; i++){
//                    for (int j=0; j<Width; j++) {
////                        byte r = (byte) (outputTensor[i * 3000 + (3*j)]);
////                        byte g = (byte) (outputTensor[i * 3000 + (3*j)+1]);
////                        byte b = (byte) (outputTensor[i * 3000 + (3*j)+2]);
//                        byte r = (byte) (outputTensor[(0*Height*Width)+i*Width + j]);
//                        byte g = (byte) (outputTensor[(1*Height*Width)+i*Width + j]);
//                        byte b = (byte) (outputTensor[(2*Height*Width)+i*Width + j]);
//                        outputArray2[(i*4*Width)+(4*j)+0] = r;
//                        outputArray2[(i*4*Width)+(4*j)+1] = g;
//                        outputArray2[(i*4*Width)+(4*j)+2] = b;
//                        outputArray2[(i*4*Width)+(4*j)+3] = -1;
//
//                    }
//                }

//                randomBitmap.copyPixelsFromBuffer(ByteBuffer.wrap(outputArray2));
//                mImageView.setImageBitmap(randomBitmap);


//
//            data[0] = (byte) (width & 0xFF);
//            data[1] = (byte) ((width >> 8) & 0xFF);


//            for(int i=0;i<Height;i++) {
//                for(int j=0;j<Width;j++) {
//                    int r = (int)(outputTensor[i*Height*Width+j*Channel+0]*(1024-64));
//                    int g1 = (int)(outputTensor[i*Height*Width+j*Channel+1]*(1024-64));
//                    int b = (int)(outputTensor[i*Height*Width+j*Channel+2]*(1024-64));
//                    int g2 = (int)(outputTensor[i*Height*Width+j*Channel+3]*(1024-64));
//
//                    imageBytes[i*3000*4000+j*4000]=0;
//
//                }
//            }


//            ByteBuffer bb = probabilityBuffer.getBuffer();
////            byte[] saveArray = outBuffer.array();
//            byte[] saveArray = bb.array();


//            for(int i=0;i<Height;i++) {
//                for(int j=0;j<Width;j++) {
//                    imageBytes[(i*2)*4000*2+(j*2)*2+0] = saveArray[i*Width*2+j*2+1];
//                    imageBytes[(i*2)*4000*2+(j*2)*2+1] = saveArray[i*Width*2+j*2+0];
//
//                }
//            }
//            Log.e("error", "saveArray len is"+saveArray.length);
//            outputTensor = input.getFloatArray();

//            try {
//                    outputTensor = probabilityBuffer.getFloatArray();
//                    byte[] saveArray = new byte[1*Channel*Height*Width*2];
//                    for(int i=0;i<outputTensor.length;i++){
//                        float out = outputTensor[i];
//                        out = (float) Math.pow(out, 2.22);
//                        int tmp = (int) (out*(1024-64)+64);
////                        Log.e("error", "tmp is "+ outputTensor[i]);
//                        byte tmp0 = (byte) (tmp & 0xFF);
//                        byte tmp1 = (byte) ((tmp >> 8) & 0xFF);
////                    byte tmp = (byte) (outputTensor[i]*255);
//                        saveArray[(2*i)+0]=tmp1;
//                        saveArray[(2*i)+1]=tmp0;
//                    }
//                    File rawsave = new File(Environment.
//                            getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM),
//                            "TMP2" + ".dng");
//                    if (!rawsave.exists()) {
//                        rawsave.createNewFile();
//                    }
//                    FileOutputStream fos = new FileOutputStream(rawsave);
//                    fos.write(saveArray);
//                    fos.close();
//                } catch (Exception e) {
//                    Log.e(TAG, e.getMessage());
//                }


                for (int i = 0; i < Height; i++) {
                    for (int j = 0; j < Width; j++) {

                        float r = outputTensor[i * Width * Channel + j * Channel + 0];
                        float g1 = outputTensor[i * Width * Channel + j * Channel + 1];
                        float b = outputTensor[i * Width * Channel + j * Channel + 2];
                        float g2 = outputTensor[i * Width * Channel + j * Channel + 3];

                        r = (float) Math.pow(customReLU(r), 2.22);
                        g1 = (float) Math.pow(customReLU(g1), 2.22);
                        b = (float) Math.pow(customReLU(b), 2.22);
                        g2 = (float) Math.pow(customReLU(g2), 2.22);


                        int tmpr = (int) (r * (1024 - 64) + 64);
                        byte tmpr_0 = (byte) (tmpr & 0xFF);
                        byte tmpr_1 = (byte) ((tmpr >> 8) & 0xFF);

                        int tmpg1 = (int) (g1 * (1024 - 64) + 64);
                        byte tmpg1_0 = (byte) (tmpg1 & 0xFF);
                        byte tmpg1_1 = (byte) ((tmpg1 >> 8) & 0xFF);

                        int tmpb = (int) (b * (1024 - 64) + 64);
                        byte tmpb_0 = (byte) (tmpb & 0xFF);
                        byte tmpb_1 = (byte) ((tmpb >> 8) & 0xFF);

                        int tmpg2 = (int) (g2 * (1024 - 64) + 64);
                        byte tmpg2_0 = (byte) (tmpg2 & 0xFF);
                        byte tmpg2_1 = (byte) ((tmpg2 >> 8) & 0xFF);

                        imageBytes[(i * 2) * 4000 * 2 + (j * 2) * 2 + 0] = tmpg1_0;
                        imageBytes[(i * 2) * 4000 * 2 + (j * 2) * 2 + 1] = tmpg1_1;


                        imageBytes[(i * 2) * 4000 * 2 + (j * 2 + 1) * 2 + 0] = tmpb_0;
                        imageBytes[(i * 2) * 4000 * 2 + (j * 2 + 1) * 2 + 1] = tmpb_1;
//
                        imageBytes[(i * 2 + 1) * 4000 * 2 + (j * 2) * 2 + 0] = tmpr_0;
                        imageBytes[(i * 2 + 1) * 4000 * 2 + (j * 2) * 2 + 1] = tmpr_1;
//
                        imageBytes[(i * 2 + 1) * 4000 * 2 + (j * 2 + 1) * 2 + 0] = tmpg2_0;
                        imageBytes[(i * 2 + 1) * 4000 * 2 + (j * 2 + 1) * 2 + 1] = tmpg2_1;


//                    }
                    }
                }
//                for(int i=0;i<Height*Width*Channel;i++){
//                    int tmp = (int) (outputTensor[i]*1024);
//                    byte tmp0 = (byte) (tmp & 0xFF);
//                    byte tmp1 = (byte) ((tmp >> 8) & 0xFF);
////                    Log.e("error", "tmp is "+outputTensor[i]);
//                    imageBytes[2*i]=tmp1;
//                    imageBytes[(2*i)+1]=tmp0;
//
//                }
//            }

//                byte[] II = new byte[4000*3000*4];
//                for (int i=0; i< Height; i++){
//                    for (int j=0; j<Width; j++) {
//                        II[(i*4*Width)+(4*j)+0] = outputArray2[(i*4*Width)+(4*j)+0];
//                        II[(i*4*Width)+(4*j)+1] = outputArray2[(i*4*Width)+(4*j)+1];
//                        II[(i*4*Width)+(4*j)+2] = outputArray2[(i*4*Width)+(4*j)+2];
//                        II[(i*4*Width)+(4*j)+3] = outputArray2[(i*4*Width)+(4*j)+1];
//
//                    }
//                }

//                Random rd = new Random();
//                rd.nextBytes(II);

                InputStream targetStream = new ByteArrayInputStream(imageBytes);
                File rawFile = new File(Environment.
                        getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM),
                        "RAW_" + generateTimestamp() + ".dng");
                Rect mRect = mCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE);

                DngCreator dngCreator = new DngCreator(mCharacteristics, result);
                FileOutputStream output = null;

                try {
                    output = new FileOutputStream(rawFile);
                    dngCreator.writeInputStream(output, largestRaw, targetStream, 0);
                } catch (IOException e) {
                    e.printStackTrace();
                }


                MediaScannerConnection.scanFile(getContext(), new String[]{rawFile.getPath()},
                        /*mimeTypes*/null, new MediaScannerConnection.MediaScannerConnectionClient() {
                            @Override
                            public void onMediaScannerConnected() {
                                // Do nothing
                            }

                            @Override
                            public void onScanCompleted(String path, Uri uri) {
                                Log.i(TAG, "Scanned " + path + ":");
                                Log.i(TAG, "-> uri=" + uri);
                            }
                        });

                Log.e("error", "Finished ever");
//            Log.e("error", String.valueOf(tflite.getOutputIndex((String)"output")));
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
            }
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
        AssetFileDescriptor fileDescriptor= getResources().getAssets().openFd("model_float32.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }
    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        Log.e("error", "onViewCreated2");
//        List<byte[]> array = new ArrayList<>();
//        for(int i=0;i<5;i++){
//            array.add(new byte[3000*4000*2]);
//
//        }
        try {
            int numThreads = 4;
            CompatibilityList compatList = new CompatibilityList();

            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            Interpreter.Options tfLiteOptions = new Interpreter.Options();
            tfLiteOptions.setNumThreads(numThreads);
//            tfLiteOptions.setUseNNAPI(true);
            tfLiteOptions.addDelegate(gpuDelegate);
            MappedByteBuffer tfliteModel;
            tfliteModel = loadModelFile();
            tflite = new Interpreter(tfliteModel, tfLiteOptions);
        } catch (IOException e) {
            e.printStackTrace();
        }


        view.findViewById(R.id.picture).setOnClickListener(this);
//        view.findViewById(R.id.info).setOnClickListener(this);
        mTextureView = (AutoFitTextureView) view.findViewById(R.id.texture);
//        mImageView = (ImageView) view.findViewById(R.id.imageView);
        mTextureViewAutoExp = view.findViewById(R.id.textViewAutoExp);
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


                largestRaw = Collections.max(
                        Arrays.asList(map.getOutputSizes(ImageFormat.RAW_SENSOR)),
                        new CompareSizesByArea());

                imageBytes = new byte[largestRaw.getHeight()*largestRaw.getWidth()*2];
                synchronized (mCameraStateLock) {
                    // Set up ImageReaders for JPEG and RAW outputs.  Place these in a reference
                    // counted wrapper to ensure they are only closed when all background tasks
                    // using them are finished.

                    if (mRawImageReader == null || mRawImageReader.getAndRetain() == null) {
                        mRawImageReader = new RefCountedAutoCloseable<>(
                                ImageReader.newInstance(largestRaw.getWidth(),
                                        largestRaw.getHeight(), ImageFormat.RAW_SENSOR, /*maxImages*/ 5));
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
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            // We configure the size of default buffer to be the size of camera preview we want.
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());

            // This is the output Surface we need to start preview.
            Surface surface = new Surface(texture);

            // We set up a CaptureRequest.Builder with the output Surface.
            mPreviewRequestBuilder
                    = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mPreviewRequestBuilder.addTarget(surface);
            // Here, we create a CameraCaptureSession for camera preview.
            mCameraDevice.createCaptureSession(Arrays.asList(surface,
                            mRawImageReader.get().getSurface()), new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                            synchronized (mCameraStateLock) {
                                // The camera is already closed
                                if (null == mCameraDevice) {
                                    return;
                                }
                                try {

//                                    setup3AControlsLocked(mPreviewRequestBuilder);
                                    // Finally, we start displaying the camera preview.
                                    cameraCaptureSession.setRepeatingRequest(
                                            mPreviewRequestBuilder.build(),
                                            mPreviewCallback, mBackgroundHandler);
//                                    mState = STATE_PREVIEW;
                                } catch (CameraAccessException | IllegalStateException e) {
                                    e.printStackTrace();
                                    return;
                                }
                                // When the session is ready, we start displaying the preview.
                                mCaptureSession = cameraCaptureSession;
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {
                            showToast("Failed to configure camera.");
                        }
                    }, mBackgroundHandler
            );
            setSeekBarShutterSpeed();
            setSeekBarISO();
//        captureStillPictureLocked();
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
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
        synchronized (mCameraStateLock) {
//        mState = STATE_WAITING_FOR_3A_CONVERGENCE;
            try {

                // This is the CaptureRequest.Builder that we use to take a picture.
//            final CaptureRequest.Builder captureBuilder =

                captureBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
                captureBuilder.addTarget(mRawImageReader.get().getSurface());
                captureBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
                captureBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                captureBuilder.set(CaptureRequest.SENSOR_SENSITIVITY, mISO);
                captureBuilder.set(CaptureRequest.SENSOR_EXPOSURE_TIME, mShutterSpeed);//(long)(0.02 *1000000000));
                // Use the same AE and AF modes as the preview.
//            setup3AControlsLocked(captureBuilder);
                // Set request tag to easily track results in callbacks.
//            captureBuilder.setTag(mRequestCounter.getAndIncrement());
//            mCaptureSession.setRepeatingRequest(captureBuilder.build(), mCaptureCallback, mBackgroundHandler);
                mCaptureRequest = captureBuilder.build();


//            List<CaptureRequest> listCaptureRequest = new ArrayList<>();
//            for(int i =0; i<1; i++){
//                listCaptureRequest.add(mCaptureRequest);
//            }
                mCaptureSession.capture(mCaptureRequest, mCaptureCallback, mBackgroundHandler);
//            mCaptureSession.capture(mCaptureRequest, mCaptureCallback, mBackgroundHandler);

//            setSeekBarShutterSpeed();
//            setSeekBarISO();
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
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
    private static String generateTimestamp() {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US);
        return sdf.format(new Date());
    }
}
