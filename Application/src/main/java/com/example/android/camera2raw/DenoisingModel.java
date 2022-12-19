package com.example.android.camera2raw;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.media.Image;
import android.util.Log;
import android.util.Size;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class DenoisingModel {

    private Context mContext;
    private Interpreter tfLiteModel;

    int Height = 1488;
    int Width = 2000;
    int Channel = 4;
    int rIdx;
    int g1Idx;
    int bIdx;
    int g2Idx;
    int[][] sliceIdx = {{0,0}, {0,1}, {1,0}, {1,1}};
    float[] inputTensor = new float[1*Channel*Height*Width];
    float[] outputTensor = new float[1*Channel*Height*Width];
    Size mLargestSize;
    byte[] imageBytes;
    TensorBuffer input = TensorBuffer.createFixedSize(new int[] {1, Height, Width, Channel}, DataType.FLOAT32);
    TensorBuffer probabilityBuffer =
            TensorBuffer.createFixedSize(new int[]{1, Height,Width,Channel}, DataType.FLOAT32);

    ByteBuffer inpBuffer;
    ByteBuffer outBuffer;

    public DenoisingModel(Context context){
        this.mContext = context;


    }
    public void initBytesArray(Size LargetSize){
        this.mLargestSize = LargetSize;
        int rawHeight = this.mLargestSize.getHeight();
        int rawWidth = this.mLargestSize.getWidth();
        this.imageBytes = new byte[rawHeight*rawWidth*2];
    }

    public Interpreter loadModelFile(String modelName, int numThreads) throws IOException {
        AssetFileDescriptor fileDescriptor=  mContext.getResources().getAssets().openFd(modelName);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        CompatibilityList compatList = new CompatibilityList();

        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLiteOptions.setNumThreads(numThreads);
        tfLiteOptions.setUseNNAPI(true);
        tfLiteOptions.addDelegate(gpuDelegate);
        MappedByteBuffer tfliteModel;
        tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
        Interpreter tflite = new Interpreter(tfliteModel, tfLiteOptions);
        this.tfLiteModel = tflite;
        return tflite;
    }
    public void setBayerPattern(String colorPattern){
        rIdx = colorPattern.indexOf("R");
        bIdx = colorPattern.indexOf("B");
        g1Idx = colorPattern.indexOf("G");
        g2Idx = colorPattern.lastIndexOf("G");
    }

    public float customReLU(float input){
        if(input<0)
            input=0;
        if (input>1)
            input=1;
        return input;
    }
    public void initTensor(Image mImage, int mRate) {

        ByteBuffer buffer = mImage.getPlanes()[0].getBuffer();
        buffer.get(imageBytes);
//        Log.e("error", "denoising init Tensor");
        for (int i = 0; i < Height * 2; i = i + 2) {
            for (int j = 0; j < Width * 2; j = j + 2) {
//
//                float g1 = (float) ((imageBytes[(i * 2 * mLargestSize.getWidth()) + (j * 2)] & 0xFF) | ((imageBytes[(i * 2 * mLargestSize.getWidth()) + (j * 2) + 1] & 0xFF) << 8));
//                float b = (float) ((imageBytes[(i * 2 * mLargestSize.getWidth()) + ((j + 1) * 2)] & 0xFF) | ((imageBytes[(i * 2 * mLargestSize.getWidth()) + ((j + 1) * 2) + 1] & 0xFF) << 8));
//                float r = (float) ((imageBytes[((i + 1) * 2 * mLargestSize.getWidth()) + (j * 2)] & 0xFF) | ((imageBytes[((i + 1) * 2 * mLargestSize.getWidth()) + (j * 2) + 1] & 0xFF) << 8));
//                float g2 = (float) ((imageBytes[((i + 1) * 2 * mLargestSize.getWidth()) + ((j + 1) * 2)] & 0xFF) | ((imageBytes[((i + 1) * 2 * mLargestSize.getWidth()) + ((j + 1) * 2) + 1] & 0xFF) << 8));
                float r = (float) ((imageBytes[((i + sliceIdx[rIdx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[rIdx][1]) * 2)] & 0xFF) |
                                  ((imageBytes[((i + sliceIdx[rIdx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[rIdx][1]) * 2) + 1] & 0xFF) << 8));

                float g1 = (float) ((imageBytes[((i + sliceIdx[g1Idx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[g1Idx][1]) * 2)] & 0xFF) |
                        ((imageBytes[((i + sliceIdx[g1Idx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[g1Idx][1]) * 2) + 1] & 0xFF) << 8));

                float b = (float) ((imageBytes[((i + sliceIdx[bIdx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[bIdx][1]) * 2)] & 0xFF) |
                        ((imageBytes[((i + sliceIdx[bIdx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[bIdx][1]) * 2) + 1] & 0xFF) << 8));

                float g2 = (float) ((imageBytes[((i + sliceIdx[g2Idx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[g2Idx][1]) * 2)] & 0xFF) |
                        ((imageBytes[((i + sliceIdx[g2Idx][0]) * 2 * mLargestSize.getWidth()) + ((j+sliceIdx[g2Idx][1]) * 2) + 1] & 0xFF) << 8));


                g1 = mRate * (g1 - 64) / (1024 - 64);
                r = mRate * (r - 64) / (1024 - 64);
                b = mRate * (b - 64) / (1024 - 64);
                g2 = mRate * (g2 - 64) / (1024 - 64);

                g1 = (float) Math.pow(customReLU(g1), 1 / 2.22);
                r = (float) Math.pow(customReLU(r), 1 / 2.22);
                b = (float) Math.pow(customReLU(b), 1 / 2.22);
                g2 = (float) Math.pow(customReLU(g2), 1 / 2.22);

                inputTensor[(i / 2) * Width * Channel + (j / 2) * Channel + 0] = r;
                inputTensor[(i / 2) * Width * Channel + (j / 2) * Channel + 1] = g1;
                inputTensor[(i / 2) * Width * Channel + (j / 2) * Channel + 2] = b;
                inputTensor[(i / 2) * Width * Channel + (j / 2) * Channel + 3] = g2;
            }
        }

        this.input.loadArray(inputTensor);
        inpBuffer = input.getBuffer();
        outBuffer = probabilityBuffer.getBuffer();
    }

    public byte[] floatArray2ByteArray(float[] floatArray, byte[] byteArray, int Height, int Width, int Channel){
//        Log.e("error", "denoising floatArray2ByteArray");
        int rawHeight = this.mLargestSize.getHeight();
        int rawWidth = this.mLargestSize.getWidth();

        for (int i = 0; i < Height; i++) {
            for (int j = 0; j < Width; j++) {

                float r = floatArray[i * Width * Channel + j * Channel + 0];
                float g1 = floatArray[i * Width * Channel + j * Channel + 1];
                float b = floatArray[i * Width * Channel + j * Channel + 2];
                float g2 = floatArray[i * Width * Channel + j * Channel + 3];

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

                byteArray[(i * 2 + sliceIdx[rIdx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[rIdx][1]) * 2 + 0] = tmpr_0;
                byteArray[(i * 2 + sliceIdx[rIdx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[rIdx][1]) * 2 + 1] = tmpr_1;


                byteArray[(i * 2 + sliceIdx[g1Idx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[g1Idx][1]) * 2 + 0] = tmpg1_0;
                byteArray[(i * 2 + sliceIdx[g1Idx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[g1Idx][1]) * 2 + 1] = tmpg1_1;
                //
                byteArray[(i * 2 + sliceIdx[bIdx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[bIdx][1]) * 2 + 0] = tmpb_0;
                byteArray[(i * 2 + sliceIdx[bIdx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[bIdx][1]) * 2 + 1] = tmpb_1;
                //
                byteArray[(i * 2 + sliceIdx[g2Idx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[g2Idx][1]) * 2 + 0] = tmpg2_0;
                byteArray[(i * 2 + sliceIdx[g2Idx][0]) * rawWidth * 2 + (j * 2 + sliceIdx[g2Idx][1]) * 2 + 1] = tmpg2_1;
            }
        }
        return byteArray;
    }
    public byte[] getOuputBytesArray(){

        this.tfLiteModel.run(inpBuffer, outBuffer);
        this.outputTensor = probabilityBuffer.getFloatArray();
        Arrays.fill(imageBytes, (byte) 0);
        this.imageBytes = floatArray2ByteArray(outputTensor,imageBytes,Height,Width,Channel);
//        Log.e("error", "denoising finish getOuputBytesArray");
        return this.imageBytes;
    }

}
