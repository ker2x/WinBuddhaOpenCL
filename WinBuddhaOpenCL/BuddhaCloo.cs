using System;
using System.Collections.Generic;
using System.Text;

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
    float escapeOrbit
)
{
    int iter = 0;
    float zr = 0.0;
    float zi = 0.0;
    float zr2 = 0.0;
    float zi2 = 0.0;
    float temp;

    //Quick rejection check if c is in 2nd order period bulb
    if( sqrt( ((cr+1.0) * (cr+1.0)) + (ci * ci) ) < 0.25 ) return true;

    //Quick rejection check if c is in main cardioid
    float tempi = ci*(-4.0);
    float tempr = 1.0 - cr*4.0;
    float theta = atan2(tempi, tempr)/2.0;
    float r = pow( tempr*tempr + tempi*tempi, 0.25);
    tempr = 1.0 - r * cos(theta);
    tempi = -r * sin(theta);
    if( (tempr * tempr + tempi * tempi) < 1.0) return true;

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
    const float  escapeOrbit,
    __global float* randomXBuffer,
    __global float* randomYBuffer,
    __global uint*  outputBuffer)
{
    const int xId = get_global_id(0);
    const int yId = get_global_id(1);

    const float deltaReal = (realMax - realMin);
    const float deltaImaginary = (imaginaryMax - imaginaryMin);

    float cr = realMin + randomXBuffer[xId] * deltaReal ;
    float ci = imaginaryMin + randomYBuffer[yId] * deltaImaginary ;

    int iter   = 0;
    float zr   = 0.0;
    float zi   = 0.0;
    float zr2  = 0.0;
    float zi2  = 0.0;
    float temp = 0.0;

    int x,y;

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
                outputBuffer[x + (y * width)] += 1;
                //outputBuffer[(y * width) + x] = 10;

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

        //Cloo
        public ComputePlatform clPlatform;
        public ComputeContext clContext;
        public ComputeContextPropertyList clProperties;
        public ComputeKernel clKernel_buddhabrot;
        public ComputeKernel clKernel_xorshift;
        public ComputeProgram clProgram;
        public ComputeCommandQueue clCommands;
        public ComputeEventList clEvents;

        public ComputeBuffer<float> d_randomXBuffer;
        public ComputeBuffer<float> d_randomYBuffer;
        public ComputeBuffer<uint>  d_outputBuffer;

        public uint[] h_outputBuffer;
        public float[] h_randomX, h_randomY;
        GCHandle gc_outputBuffer, gc_randomX, gc_randomY;

        static int workSize = 1000;

        uint seed1;
        uint seed2;
        uint seed3;
        uint seed4;
        Random R;

        //fractal
        float realMin, realMax, imaginaryMin, imaginaryMax, escapeOrbit;
        public int minIter, maxIter, width, height;



        public BuddhaCloo()
        {
            clPlatform = ComputePlatform.Platforms[0];
            clProperties = new ComputeContextPropertyList(clPlatform);
            clContext = new ComputeContext(clPlatform.Devices, clProperties, null, IntPtr.Zero);
            clCommands = new ComputeCommandQueue(clContext, clContext.Devices[0], ComputeCommandQueueFlags.None);
            clEvents = new ComputeEventList();
            clProgram = new ComputeProgram(clContext, new string[] { kernelSource });

            

            R = new Random();
            seed1 = 123456789;
            seed2 = 362436069;
            seed3 = 521288629;
            seed4 = 88675123;

            realMin = -2.25f;
            realMax = 0.75f;
            imaginaryMin = -1.5f;
            imaginaryMax = 1.5f;
            minIter = 100;
            maxIter = 1000;
            width = 600;
            height = 600;
            escapeOrbit = 4.0f;
            h_outputBuffer = new uint[width * height];
            gc_outputBuffer = GCHandle.Alloc(h_outputBuffer, GCHandleType.Pinned);

            h_randomX = new float[workSize];
            gc_randomX = GCHandle.Alloc(h_randomX, GCHandleType.Pinned);

            h_randomY = new float[workSize];
            gc_randomY = GCHandle.Alloc(h_randomY, GCHandleType.Pinned);
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
            d_outputBuffer  = new ComputeBuffer<uint>(clContext, ComputeMemoryFlags.ReadWrite, width*height);
        }

        public void ConfigureKernel()
        {
            /*__kernel void xorshift(
            uint s1,
            uint s2,
            uint s3,
            uint s4,
            const int bufferSize,
            __global float* randomXBuffer
            __global float* randomYBuffer
            )
            */

            clKernel_xorshift.SetValueArgument<uint>(0, seed1);
            clKernel_xorshift.SetValueArgument<uint>(1, seed2);
            clKernel_xorshift.SetValueArgument<uint>(2, seed3);
            clKernel_xorshift.SetValueArgument<uint>(3, seed4);
            clKernel_xorshift.SetValueArgument<int>(4, workSize);
            clKernel_xorshift.SetMemoryArgument(5, d_randomXBuffer);
            clKernel_xorshift.SetMemoryArgument(6, d_randomYBuffer);

            /*
            __kernel void buddhabrot(
            const float realMin,
            const float realMax,
            const float imaginaryMin,
            const float imaginaryMax,
            const uint  minIter,
            const uint  maxIter,
            const uint  width,
            const uint  height,
            const float  escapeOrbit,
            __global float* randomXBuffer,
            __global float* randomYBuffer,
            __global uint*  outputBuffer            */
            clKernel_buddhabrot.SetValueArgument<float>(0, realMin);
            clKernel_buddhabrot.SetValueArgument<float>(1, realMax);
            clKernel_buddhabrot.SetValueArgument<float>(2, imaginaryMin);
            clKernel_buddhabrot.SetValueArgument<float>(3, imaginaryMax);
            clKernel_buddhabrot.SetValueArgument<uint>(4, (uint)minIter);
            clKernel_buddhabrot.SetValueArgument<uint>(5, (uint)maxIter);
            clKernel_buddhabrot.SetValueArgument<uint>(6, (uint)width);
            clKernel_buddhabrot.SetValueArgument<uint>(7, (uint)height);
            clKernel_buddhabrot.SetValueArgument<float>(8, escapeOrbit);
            clKernel_buddhabrot.SetMemoryArgument(9, d_randomXBuffer);
            clKernel_buddhabrot.SetMemoryArgument(10, d_randomYBuffer);
            clKernel_buddhabrot.SetMemoryArgument(11, d_outputBuffer);
        }

        public void ExecuteKernel_xorshift()
        {
            R = new Random();
            //for (int i = 0; i < workSize; i++)
            //{
            //    h_randomX[i] = (float)R.Next() / int.MaxValue;
            //    h_randomY[i] = (float)R.Next() / int.MaxValue;
            //}
            //clCommands.Write<float>(d_randomXBuffer, h_randomX, clEvents);
            //clCommands.Write<float>(d_randomYBuffer, h_randomY, clEvents);
            //clCommands.Finish();

            clKernel_xorshift.SetValueArgument<uint>(0, (uint)R.Next(1,int.MaxValue));
            clKernel_xorshift.SetValueArgument<uint>(1, (uint)R.Next(1,int.MaxValue));
            clKernel_xorshift.SetValueArgument<uint>(2, (uint)R.Next(1,int.MaxValue));
            clKernel_xorshift.SetValueArgument<uint>(3, (uint)R.Next(1,int.MaxValue));
            clCommands.Execute(clKernel_xorshift, null, new long[] { 1 }, null, clEvents);
        }

        public void ExecuteKernel_buddhabrot()
        {
            clCommands.Execute(clKernel_buddhabrot, null, new long[] { workSize, workSize }, null, clEvents);
            //clCommands.Finish();
        }


        public void ReadResult()
        {
            clCommands.Read(d_randomXBuffer, false, 0, workSize, gc_randomX.AddrOfPinnedObject(), clEvents);
            clCommands.Read(d_randomYBuffer, false, 0, workSize, gc_randomY.AddrOfPinnedObject(), clEvents);
            clCommands.Read(d_outputBuffer, true, 0, width * height, gc_outputBuffer.AddrOfPinnedObject(), clEvents);
            clCommands.Finish();

            //for (int i = 0; i < workSize; i++)
            //{
            //    if(h_randomX[i] != (uint)0)  Console.WriteLine( (realMin +  h_randomX[i] * (realMax - realMin)));
            //}

        }

    }
}
