import welcome_gui
import wx
class MAIN(welcome_gui.Welcome):
    None
app=wx.App()
frame=MAIN(None)
frame.Show()
app.MainLoop()
