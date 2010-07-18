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
        uint[] pixelBuffer;

        public BuddhaForm()
        {
            buddhaCloo = new BuddhaCloo();
            this.Text = "OpenCL Buddhabrot";
            this.Size = new Size(buddhaCloo.width, buddhaCloo.height);
            this.SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.Opaque | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint, true);
            backBuffer = new Bitmap(buddhaCloo.width, buddhaCloo.height);
            this.BackgroundImage = backBuffer;

            pixelBuffer = new uint[buddhaCloo.width * buddhaCloo.height];

            buddhaCloo.BuildKernels();
            buddhaCloo.AllocateBuffers();
            buddhaCloo.ConfigureKernel();
            buddhaCloo.ExecuteKernel_xorshift();
            buddhaCloo.ExecuteKernel_buddhabrot();
            buddhaCloo.ReadResult();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            buddhaCloo.ExecuteKernel_xorshift();
            buddhaCloo.ExecuteKernel_buddhabrot();
            buddhaCloo.ReadResult();
            int maxfound = 0;
            for (int i = 0; i < buddhaCloo.width * buddhaCloo.height; i++)
            {
                pixelBuffer[i] += buddhaCloo.h_outputBuffer[i];
                if (pixelBuffer[i] > maxfound) { maxfound = (int)pixelBuffer[i]; }
            }
            Console.WriteLine(maxfound);

            BitmapData Locked = backBuffer.LockBits(
                new Rectangle(0, 0, buddhaCloo.width, buddhaCloo.height),
                ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb
                );

            int[] buffer = new int[buddhaCloo.width * buddhaCloo.height];
            Marshal.Copy(Locked.Scan0, buffer, 0, buffer.Length);

            for (int i = 0; i < buddhaCloo.width * buddhaCloo.height; i++)
            {
                pixelBuffer[i] = (uint)((float)pixelBuffer[i] / (maxfound + 1) * 255.0);
                int color = (int)pixelBuffer[i];
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
