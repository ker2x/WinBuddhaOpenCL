using System;
using System.Collections.Generic;
using System.Text;

using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using Cloo;

namespace WinBuddhaOpenCL
{
    public partial class BuddhaForm : Form
    {

        BuddhaCloo buddhaCloo;
        Bitmap backBuffer;
        int[] buffer;
        int color;

        DateTime oldDate, currentDate;
        TimeSpan timeInterval;

        public BuddhaForm()
        {
            buddhaCloo = new BuddhaCloo();
            this.Text = "OpenCL Buddhabrot";
            this.Size = new Size(buddhaCloo.width, buddhaCloo.height);
            this.SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.Opaque | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint, true);
            backBuffer = new Bitmap(buddhaCloo.width, buddhaCloo.height);
            this.BackgroundImage = backBuffer;

            //pixelBuffer = new uint[buddhaCloo.width * buddhaCloo.height];
            buffer = new int[buddhaCloo.width * buddhaCloo.height];

            buddhaCloo.BuildKernels();
            buddhaCloo.AllocateBuffers();
            buddhaCloo.ConfigureKernel();
            oldDate = DateTime.Now;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            buddhaCloo.ExecuteKernel_xorshift();
            buddhaCloo.ExecuteKernel_buddhabrot();
            buddhaCloo.ReadResult();

            currentDate = DateTime.Now;
            timeInterval = currentDate - oldDate;
            oldDate = currentDate;
            Console.WriteLine("{0} samples/s at {1} iterations", (uint)(BuddhaCloo.workSize*BuddhaCloo.workSize / ((timeInterval.Seconds*1000 + timeInterval.Milliseconds)/1000.0)), buddhaCloo.maxIter );

            int maxfound = 0;
            for (int i = 0; i < buddhaCloo.width * buddhaCloo.height; i++)
            {
                if (buddhaCloo.h_outputBuffer[i] > maxfound) { maxfound = (int)buddhaCloo.h_outputBuffer[i]; }
            }

            BitmapData Locked = backBuffer.LockBits(
                new Rectangle(0, 0, buddhaCloo.width, buddhaCloo.height),
                ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb
                );

            Marshal.Copy(Locked.Scan0, buffer, 0, buffer.Length);

            for (int i = 0; i < buddhaCloo.width * buddhaCloo.height; i++)
            {
                color = (int)((Math.Sqrt(buddhaCloo.h_outputBuffer[i])) / Math.Sqrt(maxfound) * 255.0);
                buffer[i] = (((color & 0xFF) << 16) | ((color & 0xFF) << 8) | (color & 0xFF));
            }
            Marshal.Copy(buffer, 0, Locked.Scan0, buffer.Length);
            backBuffer.UnlockBits(Locked);
            e.Graphics.DrawImageUnscaled(backBuffer, 0, 0);

            this.Invalidate();
        }

        protected override void OnSizeChanged(EventArgs e)
        {
            base.OnSizeChanged(e);
        }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            base.OnKeyDown(e);
        }
    
    }

}
