using System;
using System.Windows.Forms;


namespace WinBuddhaOpenCL
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            using (BuddhaForm buddha = new BuddhaForm())
            {
                buddha.Show();
                Application.Run(buddha);
            }

        }
    }
}
