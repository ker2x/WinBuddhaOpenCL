using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using Cloo;

namespace WinBuddhaOpenCL
{
    class BuddhaCloo
    {

        public static string kernelSource= @"

//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

//Check if choosen point is in MSet
bool isInMSet(
    float cr,
    float ci,
    const uint maxIter,
    const float escapeOrbit)
{
    int iter = 0;
    float zr = 0.0;
    float zi = 0.0;
    float ci2 = ci*ci;
    float temp;

    //Quick rejection check if c is in 2nd order period bulb
    if( (cr+1.0) * (cr+1.0) + ci2 < 0.0625) return true;

    //Quick rejection check if c is in main cardioid
    float q = (cr-0.25)*(cr-0.25) + ci2;
    if( q*(q+(cr-0.25)) < 0.25*ci2) return true; 


    // test for the smaller bulb left of the period-2 bulb
    if (( ((cr+1.309)*(cr+1.309)) + ci*ci) < 0.00345) return true;

    // check for the smaller bulbs on top and bottom of the cardioid
    if ((((cr+0.125)*(cr+0.125)) + (ci-0.744)*(ci-0.744)) < 0.0088) return true;
    if ((((cr+0.125)*(cr+0.125)) + (ci+0.744)*(ci+0.744)) < 0.0088) return true;

    while( (iter < maxIter) && ((zr*zr+zi*zi) < escapeOrbit) )
    {
        temp = zr * zi;
        zr = zr*zr - zi*zi + cr;
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
    __global float2* randomXYBuffer,
    __global uint4*  outputBuffer)
{
    float2 rand = randomXYBuffer[get_global_id(0)];    

    const float deltaReal = (realMax - realMin);
    const float deltaImaginary = (imaginaryMax - imaginaryMin);

    //mix(a,b,c) = a + (b-a)*c //(c must be in the range 0.0 ... 1.0
    //float cr = realMin + rand.x * deltaReal ;
    //float cr = realMin + (realMax - realMin) * rand.x ;
    //float ci = imaginaryMin + rand.y * deltaImaginary ;
    float cr = mix(realMin, realMax, rand.x);
    float ci = mix(imaginaryMin, imaginaryMax, rand.y);

    int x, y;
    int iter   = 0;
    float zr   = 0.0;
    float zi   = 0.0;
    float temp = 0.0;


    if( isInMSet(cr,ci, maxIter, escapeOrbit) == false)
    {    
        while( (iter < maxIter) && ((zr*zr+zi*zi) < escapeOrbit) )
        {
            temp = zr * zi;
            zr = zr*zr - zi*zi + cr;
            zi = temp + temp + ci;

            x = ((width) * (zr - realMin) / deltaReal);
            y = ((height) * (zi - imaginaryMin) / deltaImaginary);

            if( (iter > minIter) && (x>0) && (y>0) && (x<width) && (y<height) )
            {
                if( (iter > minColor.x) && (iter < maxColor.x) ) { outputBuffer[x + (y * width)].x++; }
                if( (iter > minColor.y) && (iter < maxColor.y) ) { outputBuffer[x + (y * width)].y++; }
                if( (iter > minColor.z) && (iter < maxColor.z) ) { outputBuffer[x + (y * width)].z++; }
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
    __global float2* randomXYBuffer
)
{
    uint st;
    float2 tmp;

    for(int i=0; i < bufferSize; i++)
    {
        st = s1 ^ (s1 << 11);
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s4 ^ (s4 >> 19) ^ ( st ^ (st >> 18));
        tmp.x = (float)s4 / UINT_MAX;

        st = s1 ^ (s1 << 11);
        s1 = s2;
        s2 = s3;
        s3 = s4;
        s4 = s4 ^ (s4 >> 19) ^ ( st ^ (st >> 18));
        tmp.y = (float)s4 / UINT_MAX;
        randomXYBuffer[i] = tmp;

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

        public struct VectorFloatXY
        {
            public uint X;
            public uint Y;
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

        public ComputeBuffer<VectorFloatXY> d_randomXYbuffer;
        public ComputeBuffer<ColorVectorRGBA>  d_outputBuffer;

        public ColorVectorRGBA[] h_outputBuffer;

        public static int workSize = 1000000;

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

            /*
            realMin = -1.05f;
            realMax = -0.9f;
            imaginaryMin = -0.3f;
            imaginaryMax = -0.225f;

            minIter = 20000;
            maxIter = 200000;
            escapeOrbit = 4.0f;

            minColor.R = 20000;
            maxColor.R = 60000;

            minColor.G = 60000;
            maxColor.G = 100000;
            
            minColor.B = 100000;
            maxColor.B = 200000;
*/

            realMin = -1.22f;
            realMax = -1.0f;
            imaginaryMin = 0.16f;
            imaginaryMax = 0.32f;
            //realMin = -1.5f;
            //realMax = 0.75f;
            //imaginaryMin = -1.5f;
            //imaginaryMax = 1.5f;

            minIter =20;
            maxIter = 1600;
            escapeOrbit = 4.0f;

            minColor.R = 20;
            maxColor.R = 400;

            minColor.G = 400;
            maxColor.G = 800;

            minColor.B = 800;
            maxColor.B = 1600;

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
            d_randomXYbuffer = new ComputeBuffer<VectorFloatXY>(clContext, ComputeMemoryFlags.ReadWrite, workSize);
            d_outputBuffer  = new ComputeBuffer<ColorVectorRGBA>(clContext, ComputeMemoryFlags.ReadWrite, width*height);
        }

        public void ConfigureKernel()
        {
            clKernel_xorshift.SetValueArgument<uint>(0, seed1);
            clKernel_xorshift.SetValueArgument<uint>(1, seed2);
            clKernel_xorshift.SetValueArgument<uint>(2, seed3);
            clKernel_xorshift.SetValueArgument<uint>(3, seed4);
            clKernel_xorshift.SetValueArgument<int>(4, workSize);
            clKernel_xorshift.SetMemoryArgument(5, d_randomXYbuffer);


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
            clKernel_buddhabrot.SetMemoryArgument(11, d_randomXYbuffer);
            clKernel_buddhabrot.SetMemoryArgument(12, d_outputBuffer);
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
            clCommands.Execute(clKernel_buddhabrot, null, new long[] { workSize }, null, clEvents);
        }


        public void ReadResult()
        {
            clCommands.Read(d_outputBuffer, true, 0, width * height, gc_outputBuffer.AddrOfPinnedObject(), clEvents);
            clCommands.Finish();
        }

    }
}
