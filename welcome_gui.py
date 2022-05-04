# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

from pickle import NONE
import wx
import wx.xrc
import frame_structure

###########################################################################
## Class Welcome
###########################################################################
class MAIN_GUI(frame_structure.main_gui):
    None
def initialize():
	None
	app1=wx.App()
	frame=MAIN_GUI(None)
	frame.Show()
	app1.MainLoop()
class Welcome ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"欢迎使用LSTM多维数据预测软件", pos = wx.DefaultPosition, size = wx.Size( 800,600 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_3DLIGHT ) )

		bSizer3 = wx.BoxSizer( wx.VERTICAL )

		self.m_bitmap3 = wx.StaticBitmap( self, wx.ID_ANY, wx.Bitmap( u"wel_bp1.jpg", wx.BITMAP_TYPE_ANY ), wx.DefaultPosition, wx.Size( 800,400 ), 0 )
		bSizer3.Add( self.m_bitmap3, 0, wx.ALIGN_CENTER|wx.ALL, 5 )

		gbSizer3 = wx.GridBagSizer( 2, 2 )
		gbSizer3.SetFlexibleDirection( wx.BOTH )
		gbSizer3.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_staticText13 = wx.StaticText( self, wx.ID_ANY, u"单击“初始化”按键，程序开始自动配置Keras环境，可能需要一段时间。", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText13.Wrap( -1 )

		gbSizer3.Add( self.m_staticText13, wx.GBPosition( 0, 0 ), wx.GBSpan( 1, 2 ), wx.ALIGN_CENTER|wx.ALL, 5 )

		self.m_button6 = wx.Button( self, wx.ID_ANY, u"初始化", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer3.Add( self.m_button6, wx.GBPosition( 1, 0 ), wx.GBSpan( 1, 1 ), wx.ALIGN_CENTER|wx.ALL, 5 )

		self.m_button7 = wx.Button( self, wx.ID_ANY, u"退出", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer3.Add( self.m_button7, wx.GBPosition( 1, 1 ), wx.GBSpan( 1, 1 ), wx.ALIGN_CENTER|wx.ALL, 5 )


		bSizer3.Add( gbSizer3, 1, wx.ALIGN_CENTER|wx.ALL|wx.RESERVE_SPACE_EVEN_IF_HIDDEN, 5 )

		self.m_staticText12 = wx.StaticText( self, wx.ID_ANY, u"运行环境：Keras 2.7.0 (Tensorflow backend), numpy 1.21.5, scikit-learn  0.22.1, scipy 1.7.3, pandas 1.3.5", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText12.Wrap( -1 )

		bSizer3.Add( self.m_staticText12, 0, wx.ALL, 5 )

		self.m_staticText15 = wx.StaticText( self, wx.ID_ANY, u"版本号：v1.0.0", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText15.Wrap( -1 )

		bSizer3.Add( self.m_staticText15, 0, wx.ALL, 5 )


		self.SetSizer( bSizer3 )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.m_button6.Bind( wx.EVT_BUTTON, self.PREPROESS )
		self.m_button7.Bind( wx.EVT_BUTTON, self.QUIT )

	def __del__( self ):
		pass


	# Virtual event handlers, override them in your derived class
	def PREPROESS( self, event ):
		self.Destroy()
		initialize()
	def QUIT( self, event ):
		self.Close(True)
