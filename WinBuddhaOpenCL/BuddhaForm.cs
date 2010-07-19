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
        int colorR, colorG, colorB;

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
            Console.WriteLine(
                "{0} samples/s at {1} iterations", 
                (uint)(BuddhaCloo.workSize*BuddhaCloo.workSize / ((timeInterval.Seconds*1000 + timeInterval.Milliseconds)/1000.0)), 
                buddhaCloo.maxIter);

            int maxfound = 0;
            int maxfoundR = 0;
            int maxfoundG = 0;
            int maxfoundB = 0;

            for (int i = 0; i < buddhaCloo.width * buddhaCloo.height; i++)
            {
                if (buddhaCloo.h_outputBuffer[i].R > maxfoundR) { maxfoundR = (int)buddhaCloo.h_outputBuffer[i].R; }
                if (buddhaCloo.h_outputBuffer[i].G > maxfoundG) { maxfoundG = (int)buddhaCloo.h_outputBuffer[i].G; }
                if (buddhaCloo.h_outputBuffer[i].B > maxfoundB) { maxfoundB = (int)buddhaCloo.h_outputBuffer[i].B; }
            }

            if (maxfoundR > maxfound) maxfound = maxfoundR;
            if (maxfoundG > maxfound) maxfound = maxfoundG;
            if (maxfoundB > maxfound) maxfound = maxfoundB;

            BitmapData Locked = backBuffer.LockBits(
                new Rectangle(0, 0, buddhaCloo.width, buddhaCloo.height),
                ImageLockMode.ReadWrite, PixelFormat.Format32bppRgb
                );

            Marshal.Copy(Locked.Scan0, buffer, 0, buffer.Length);

            for (int i = 0; i < buddhaCloo.width * buddhaCloo.height; i++)
            {
                colorR = (int)((Math.Sqrt(buddhaCloo.h_outputBuffer[i].R)) / Math.Sqrt(maxfound) * 255.0);
                colorG = (int)((Math.Sqrt(buddhaCloo.h_outputBuffer[i].G)) / Math.Sqrt(maxfound) * 255.0);
                colorB = (int)((Math.Sqrt(buddhaCloo.h_outputBuffer[i].B)) / Math.Sqrt(maxfound) * 255.0);
                buffer[i] = (((colorR & 0xFF) << 16) | ((colorG & 0xFF) << 8) | (colorB & 0xFF));
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
            if (e.KeyCode == Keys.Escape)
                Application.Exit();

            base.OnKeyDown(e);
        }
    
    }

}
