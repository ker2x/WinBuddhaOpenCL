﻿using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using Cloo;

namespace WinBuddhaOpenCL
{
    class BuddhaCloo
    {

        public static string kernelSource= @"

//Check if choosen point is in MSet
bool isInMSet(
    float cr,
    float ci,
    uint maxIter,
    float escapeOrbit)
{
    int iter = 0;
    float zr = 0.0;
    float zi = 0.0;
    float zr2 = 0.0;
    float zi2 = 0.0;
    float ci2 = ci*ci;
    float temp;

    //Quick rejection check if c is in 2nd order period bulb
    //if( sqrt( ((cr+1.0) * (cr+1.0)) + ci2 ) < 0.25 ) return true;
    if( (cr+1.0) * (cr+1.0) + ci2 < 0.0625) return true;

    //Quick rejection check if c is in main cardioid
    float q = (cr-0.25)*(cr-0.25) + ci2;
    if( q*(q+(cr-0.25)) < 0.25*ci2) return true;

    while( (iter < maxIter) && ((zr2+zi2) < escapeOrbit) )
    {
        temp = zr * zi;
        zr2 = zr * zr;
        zi2 = zi * zi;
        zr = zr2 - zi2 + cr;
        zi = temp + temp + ci;
        iter++;
    }

    if( iter < maxIter)
    {
        return false;
    } else {
        return true;
    }

}


//Main kernel
__kernel void buddhabrot(
    const float realMin,
    const float realMax,
    const float imaginaryMin,
    const float imaginaryMax,
    const uint  minIter,
    const uint  maxIter,
    const uint  width,
    const uint  height,
    const float escapeOrbit,
    const uint4 minColor,
    const uint4 maxColor,
    __global float* randomXBuffer,
    __global float* randomYBuffer,
    __global uint4*  outputBuffer)
{
    const int xId = get_global_id(0);
    const int yId = get_global_id(1);

    const float deltaReal = (realMax - realMin);
    const float deltaImaginary = (imaginaryMax - imaginaryMin);

    float cr = realMin + randomXBuffer[xId] * deltaReal ;
    float ci = imaginaryMin + randomYBuffer[yId] * deltaImaginary ;

    int x, y;
    int iter   = 0;
    float zr   = 0.0;
    float zi   = 0.0;
    float zr2  = 0.0;
    float zi2  = 0.0;
    float temp = 0.0;


    if( isInMSet(cr,ci, maxIter, escapeOrbit) == false)
    {    
        while( (iter < maxIter) && ((zr2+zi2) < escapeOrbit) )
        {
            temp = zr * zi;
            zr2 = zr * zr;
            zi2 = zi * zi;
            zr = zr2 - zi2 + cr;
            zi = temp + temp + ci;
            
            x = ((width) * (zr - realMin) / (realMax - realMin));
            y = ((height) * (zi - imaginaryMin) / (imaginaryMax - imaginaryMin));

            if( (x>0) && (y>0) && (x<width) && (y<height) && (iter > minIter))
            {
                if( (iter > minColor.x) && (iter < maxColor.x) ) outputBuffer[x + (y * width)].x += 1;
                if( (iter > minColor.y) && (iter < maxColor.y) ) outputBuffer[x + (y * width)].y += 1;
                if( (iter > minColor.z) && (iter < maxColor.z) ) outputBuffer[x + (y * width)].z += 1;
            }
            iter++;
        }
    }
}

__kernel void xorshift(
    uint s1,
    uint s2,
    uint s3,
    uint s4,
    const int bufferSize,
    __global float* randomXBuffer,
    __global float* randomYBuffer
)
{
    uint st;

    for(int i=0; i < bufferSize; i++)
    {
        st = s1 ^ (s1 << 11);
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s4 ^ (s4 >> 19) ^ ( st ^ (st >> 18));
        randomXBuffer[i] = s4 / 4294967295.0; 

        st = s1 ^ (s1 << 11);
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s4 ^ (s4 >> 19) ^ ( st ^ (st >> 18));
        randomYBuffer[i] = s4 / 4294967295.0; 
    }
}

";

        public struct ColorVectorRGBA
        {
            public uint R;
            public uint G;
            public uint B;
            public uint A;
        };

        //Cloo
        public ComputePlatform  clPlatform;
        public ComputeContext   clContext;
        public ComputeContextPropertyList clProperties;
        public ComputeKernel    clKernel_buddhabrot;
        public ComputeKernel    clKernel_xorshift;
        public ComputeProgram   clProgram;
        public ComputeCommandQueue clCommands;
        public ComputeEventList clEvents;

        public ComputeBuffer<float> d_randomXBuffer;
        public ComputeBuffer<float> d_randomYBuffer;
        public ComputeBuffer<ColorVectorRGBA>  d_outputBuffer;

        public ColorVectorRGBA[] h_outputBuffer;

        public static int workSize = 512;

        private uint seed1;
        private uint seed2;
        private uint seed3;
        private uint seed4;
        private GCHandle gc_outputBuffer;

        private Random R;

        //fractal
        public float realMin, realMax, imaginaryMin, imaginaryMax, escapeOrbit;
        public int minIter, maxIter, width, height;


        ColorVectorRGBA minColor, maxColor;


        public BuddhaCloo()
        {
            clPlatform = ComputePlatform.Platforms[0];
            clProperties = new ComputeContextPropertyList(clPlatform);
            clContext = new ComputeContext(clPlatform.Devices, clProperties, null, IntPtr.Zero);
            clCommands = new ComputeCommandQueue(clContext, clContext.Devices[0], ComputeCommandQueueFlags.None);
            clEvents = new ComputeEventList();
            clProgram = new ComputeProgram(clContext, new string[] { kernelSource });

            R = new Random();
            seed1 = (uint)R.Next();
            seed2 = (uint)R.Next();
            seed3 = (uint)R.Next();
            seed4 = (uint)R.Next();

            /* //Default buddhabrot parameters
             * realMin = -1.5f;
             * realMax = 0.75f;
             * imaginaryMin = -1.5f;
             * imaginaryMax = 1.5f;
             */

            realMin = -1.2f;
            realMax = -0.8f;
            imaginaryMin = -0.4f;
            imaginaryMax = -0.1f;

            minIter = 100;
            maxIter = 10000;
            escapeOrbit = 4.0f;

            minColor.R = 100;
            maxColor.R = 10000;

            minColor.G = 2000;
            maxColor.G = 50000;
            
            minColor.B = 5000;
            maxColor.B = 10000;


            width = 1000;
            height = 700;

            h_outputBuffer = new ColorVectorRGBA[width * height];
            gc_outputBuffer = GCHandle.Alloc(h_outputBuffer, GCHandleType.Pinned);

        }

        public void BuildKernels()
        {
            clProgram.Build(null, null, null, IntPtr.Zero);
            clKernel_buddhabrot = clProgram.CreateKernel("buddhabrot");
            clKernel_xorshift = clProgram.CreateKernel("xorshift");
        }

        public void AllocateBuffers()
        {
            d_randomXBuffer = new ComputeBuffer<float>(clContext, ComputeMemoryFlags.ReadWrite, workSize);
            d_randomYBuffer = new ComputeBuffer<float>(clContext, ComputeMemoryFlags.ReadWrite, workSize);
            d_outputBuffer  = new ComputeBuffer<ColorVectorRGBA>(clContext, ComputeMemoryFlags.ReadWrite, width*height);
        }

        public void ConfigureKernel()
        {
            clKernel_xorshift.SetValueArgument<uint>(0, seed1);
            clKernel_xorshift.SetValueArgument<uint>(1, seed2);
            clKernel_xorshift.SetValueArgument<uint>(2, seed3);
            clKernel_xorshift.SetValueArgument<uint>(3, seed4);
            clKernel_xorshift.SetValueArgument<int>(4, workSize);
            clKernel_xorshift.SetMemoryArgument(5, d_randomXBuffer);
            clKernel_xorshift.SetMemoryArgument(6, d_randomYBuffer);

            clKernel_buddhabrot.SetValueArgument<float>(0, realMin);
            clKernel_buddhabrot.SetValueArgument<float>(1, realMax);
            clKernel_buddhabrot.SetValueArgument<float>(2, imaginaryMin);
            clKernel_buddhabrot.SetValueArgument<float>(3, imaginaryMax);
            clKernel_buddhabrot.SetValueArgument<uint>(4, (uint)minIter);
            clKernel_buddhabrot.SetValueArgument<uint>(5, (uint)maxIter);
            clKernel_buddhabrot.SetValueArgument<uint>(6, (uint)width);
            clKernel_buddhabrot.SetValueArgument<uint>(7, (uint)height);
            clKernel_buddhabrot.SetValueArgument<float>(8, escapeOrbit);
            clKernel_buddhabrot.SetValueArgument<ColorVectorRGBA>(9, minColor);
            clKernel_buddhabrot.SetValueArgument<ColorVectorRGBA>(10, maxColor);
            clKernel_buddhabrot.SetMemoryArgument(11, d_randomXBuffer);
            clKernel_buddhabrot.SetMemoryArgument(12, d_randomYBuffer);
            clKernel_buddhabrot.SetMemoryArgument(13, d_outputBuffer);
        }


        public void ExecuteKernel_xorshift()
        {
            R = new Random();
            clKernel_xorshift.SetValueArgument<uint>(0, (uint)R.Next());
            clKernel_xorshift.SetValueArgument<uint>(1, (uint)R.Next());
            clKernel_xorshift.SetValueArgument<uint>(2, (uint)R.Next());
            clKernel_xorshift.SetValueArgument<uint>(3, (uint)R.Next());
            clCommands.Execute(clKernel_xorshift, null, new long[] { 1 }, null, clEvents);
        }

        public void ExecuteKernel_buddhabrot()
        {
            clCommands.Execute(clKernel_buddhabrot, null, new long[] { workSize, workSize }, null, clEvents);
        }


        public void ReadResult()
        {
            clCommands.Read(d_outputBuffer, true, 0, width * height, gc_outputBuffer.AddrOfPinnedObject(), clEvents);
            clCommands.Finish();
        }

    }
}
