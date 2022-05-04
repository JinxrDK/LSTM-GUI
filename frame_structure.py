# -*- coding: utf-8 -*-
import wx
import wx.xrc
import wx.grid
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler	
###########################################################################
## 这部分是LSTM算法的核心
###########################################################################
global model,input_data
model=Sequential()
def DATA_MAKE(DOUBLE_ENABLE,data_path):
	global scaled,scaler
	dataset=read_csv(data_path, header=0, index_col=0)
	values=dataset.values
	if DOUBLE_ENABLE==False:
		values = values.astype('float32')
	else:
		values = values.astype('float64')
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	return None
def DATA_RESHAPE(scaled,train_vol,val_vol,test_vol):
	global train_X,train_Y,val_X,val_Y,test_X,test_Y
	df=DataFrame(scaled)
	values=df.values
	n_train_data=int(train_vol*(df.shape[0]))
	n_val_data=int((train_vol+val_vol)*(df.shape[0]))
	n_test_data=int(df.shape[0]*test_vol)
	train=values[:n_train_data,:]
	val=values[n_train_data:n_val_data,:]
	test=values[n_test_data:,:]
	train_X=train[:,Y_cols:]
	train_Y=train[:,:Y_cols]
	val_X=val[:,Y_cols:]
	val_Y=val[:,:Y_cols]
	test_X=test[:,Y_cols:]
	test_Y=test[:,:Y_cols]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape,val_X.shape)
	return None
def LAYERS(model,Dense_use,layers,cells):
	model.add(LSTM(cells,activation='tanh',recurrent_activation='sigmoid',input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True))
	if layers>1:
		for i in range(layers-1):
			model.add(LSTM(cells,activation='tanh',recurrent_activation='sigmoid',return_sequences=True))
	if Dense_use==True:
		model.add(Dropout(0.5))
	model.add(Dense(Y_cols))
	return None
def MODEL_SET(optimizer,loss_function):
	model.compile(optimizer=optimizer,loss=loss_function)
	return None
def MODEL_TRAIN(lr,epochs,batch_size):
	K.set_value(model.optimizer.lr,float(lr))
	def LR_setting(epoch):
		return K.get_value(model.optimizer.lr)
	Lr_setting=LearningRateScheduler(LR_setting)
	history=model.fit(train_X,train_Y,batch_size,epochs,verbose=2,validation_data=(val_X,val_Y),shuffle=False,callbacks=[Lr_setting])
	pyplot.plot(history.history['loss'], label='train_data')
	pyplot.plot(history.history['val_loss'], label='val_data')
	pyplot.legend()
	pyplot.savefig('process.png',c="c")
def MODEL_PREDICT():
	global test_X,test_Y,Y_cols
	Yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0],test_X.shape[2]))
	Yhat=Yhat.reshape((Yhat.shape[0],Yhat.shape[2]))
	print(Yhat.shape,test_X.shape)
	inv_Yhat = np.hstack((Yhat, test_X))
	inv_Yhat = scaler.inverse_transform(inv_Yhat)
	inv_Yhat = inv_Yhat[:,:Y_cols]
	inv_Yhat = np.array(inv_Yhat)
	test_Y = test_Y.reshape((len(test_Y),Y_cols))
	inv_Y = np.hstack((test_Y, test_X))
	inv_Y = scaler.inverse_transform(inv_Y)
	inv_Y = inv_Y[:,:Y_cols]
	np.savetxt("output.csv",np.hstack((inv_Y,inv_Yhat)),delimiter=',')
	rmse_Y_cols=np.arange(Y_cols,dtype=np.float16)
	for i in range(Y_cols):
		rmse_Y_cols[i] = sqrt(mean_squared_error(inv_Y[i,:], inv_Yhat[i,:]))
	return rmse_Y_cols
###########################################################################
## 这部分是GUI框架
###########################################################################
class main_gui ( wx.Frame ):
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"基于LSTM的多维数据预测软件", pos = wx.DefaultPosition, size = wx.Size( 772,606 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_BTNFACE ) )
		gbSizer1 = wx.GridBagSizer( 3, 2 )
		gbSizer1.SetFlexibleDirection( wx.BOTH )
		gbSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"训练集设置" ), wx.VERTICAL )
		sbSizer2.SetMinSize( wx.Size( 300,300 ) )
		self.m_staticText1 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"读取训练数据", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1.Wrap( -1 )
		sbSizer2.Add( self.m_staticText1, 0, wx.ALL, 5 )
		self.m_filePicker1 = wx.FilePickerCtrl( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, u"请选择输入的数据集", u"*.csv", wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE )
		sbSizer2.Add( self.m_filePicker1, 0, wx.ALL, 5 )
		self.m_staticText2 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"读取到的数据表格式", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText2.Wrap( -1 )
		sbSizer2.Add( self.m_staticText2, 0, wx.ALL, 5 )
		self.m_grid1 = wx.grid.Grid( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_grid1.CreateGrid( 2, 1 )
		self.m_grid1.EnableEditing( False )
		self.m_grid1.EnableGridLines( True )
		self.m_grid1.EnableDragGridSize( False )
		self.m_grid1.SetMargins( 0, 0 )
		self.m_grid1.EnableDragColMove( False )
		self.m_grid1.EnableDragColSize( True )
		self.m_grid1.SetColLabelValue( 0, u"值" )
		self.m_grid1.SetColLabelSize( 30 )
		self.m_grid1.SetColLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )
		self.m_grid1.EnableDragRowSize( True )
		self.m_grid1.SetRowLabelValue( 0, u"行数" )
		self.m_grid1.SetRowLabelValue( 1, u"列数" )
		self.m_grid1.SetRowLabelValue( 2, wx.EmptyString )
		self.m_grid1.SetRowLabelSize( 80 )
		self.m_grid1.SetRowLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )
		self.m_grid1.SetLabelFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, wx.EmptyString ) )
		self.m_grid1.SetDefaultCellFont( wx.Font( 9, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL, False, "Times New Roman" ) )
		self.m_grid1.SetDefaultCellAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )
		self.m_grid1.SetMaxSize( wx.Size( 200,-1 ) )
		sbSizer2.Add( self.m_grid1, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )
		self.m_staticText3 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"设置因变量列数", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )
		sbSizer2.Add( self.m_staticText3, 0, wx.ALL, 5 )
		self.Y_ = wx.TextCtrl( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer2.Add( self.Y_, 0, wx.ALL, 5 )
		self.m_checkBox3 = wx.CheckBox( sbSizer2.GetStaticBox(), wx.ID_ANY, u"启用双精度模式处理数据", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer2.Add( self.m_checkBox3, 0, wx.ALL, 5 )
		self.m_button1 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"确认", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer2.Add( self.m_button1, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		gbSizer1.Add( sbSizer2, wx.GBPosition( 0, 0 ), wx.GBSpan( 1, 1 ), wx.EXPAND, 4 )
		sbSizer3 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"模型设置" ), wx.VERTICAL )
		sbSizer3.SetMinSize( wx.Size( 280,-1 ) )
		self.m_checkBox2 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"使用Dropout层", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer3.Add( self.m_checkBox2, 0, wx.ALL, 5 )
		self.m_staticText5 = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, u"LSTM层数", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText5.Wrap( -1 )
		sbSizer3.Add( self.m_staticText5, 0, wx.ALL, 5 )
		self.LSTM_layers = wx.TextCtrl( sbSizer3.GetStaticBox(), wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer3.Add( self.LSTM_layers, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_staticText6 = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, u"单层LSTM细胞数", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )
		sbSizer3.Add( self.m_staticText6, 0, wx.ALL, 5 )
		self.LSTM_cells = wx.TextCtrl( sbSizer3.GetStaticBox(), wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer3.Add( self.LSTM_cells, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_staticText7 = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, u"数据集划分", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText7.Wrap( -1 )
		sbSizer3.Add( self.m_staticText7, 0, wx.ALL, 5 )
		self.m_grid2 = wx.grid.Grid( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0)
		self.m_grid2.CreateGrid( 3, 1 )
		self.m_grid2.EnableEditing( True )
		self.m_grid2.EnableGridLines( True )
		self.m_grid2.EnableDragGridSize( False )
		self.m_grid2.SetMargins( 0, 0 )
		self.m_grid2.EnableDragColMove( False )
		self.m_grid2.EnableDragColSize( True )
		self.m_grid2.SetColLabelValue( 0, u"比例" )
		self.m_grid2.SetColLabelSize( 30 )
		self.m_grid2.SetColLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )
		self.m_grid2.EnableDragRowSize( True )
		self.m_grid2.SetRowLabelValue( 0, u"训练集" )
		self.m_grid2.SetRowLabelValue( 1, u"验证集" )
		self.m_grid2.SetRowLabelValue( 2, u"测试集" )
		self.m_grid2.SetRowLabelSize( 80 )
		self.m_grid2.SetRowLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )
		self.m_grid2.SetDefaultCellAlignment( wx.ALIGN_CENTER, wx.ALIGN_TOP )
		sbSizer3.Add( self.m_grid2, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_button2 = wx.Button( sbSizer3.GetStaticBox(), wx.ID_ANY, u"确认", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer3.Add( self.m_button2, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		gbSizer1.Add( sbSizer3, wx.GBPosition( 0, 1 ), wx.GBSpan( 1, 1 ), wx.EXPAND, 5 )
		sbSizer31 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"训练设置" ), wx.VERTICAL )
		sbSizer31.SetMinSize( wx.Size( 170,-1 ) )
		m_radioBox5Choices = [ u"SGD", u"RMSprop", u"Adam" ]
		self.m_radioBox5 = wx.RadioBox( sbSizer31.GetStaticBox(), wx.ID_ANY, u"优化器选择", wx.DefaultPosition, wx.DefaultSize, m_radioBox5Choices, 1, wx.RA_SPECIFY_COLS )
		self.m_radioBox5.SetSelection( 0 )
		sbSizer31.Add( self.m_radioBox5, 0, wx.ALL, 5 )
		self.m_staticText9 = wx.StaticText( sbSizer31.GetStaticBox(), wx.ID_ANY, u"学习率", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText9.Wrap( -1 )
		sbSizer31.Add( self.m_staticText9, 0, wx.ALL, 5 )
		self.Learing_rate = wx.TextCtrl( sbSizer31.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer31.Add( self.Learing_rate, 0, wx.ALL, 5 )
		self.m_staticText10 = wx.StaticText( sbSizer31.GetStaticBox(), wx.ID_ANY, u"迭代次数", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText10.Wrap( -1 )
		sbSizer31.Add( self.m_staticText10, 0, wx.ALL, 5 )
		self.m_textCtrl5 = wx.TextCtrl( sbSizer31.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer31.Add( self.m_textCtrl5, 0, wx.ALL, 5 )
		self.epoch_size = wx.StaticText( sbSizer31.GetStaticBox(), wx.ID_ANY, u"单次迭代的数据量", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.epoch_size.Wrap( -1 )
		sbSizer31.Add( self.epoch_size, 0, wx.ALL, 5 )
		self.m_textCtrl6 = wx.TextCtrl( sbSizer31.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer31.Add( self.m_textCtrl6, 0, wx.ALL, 5 )
		m_radioBox4Choices = [ u"均方误差MSE", u"均方对数误差MSLE", u"均绝对误差MAE" ]
		self.m_radioBox4 = wx.RadioBox( sbSizer31.GetStaticBox(), wx.ID_ANY, u"损失函数设置", wx.DefaultPosition, wx.DefaultSize, m_radioBox4Choices, 1, wx.RA_SPECIFY_COLS )
		self.m_radioBox4.SetSelection( 0 )
		sbSizer31.Add( self.m_radioBox4, 0, wx.ALL, 5 )
		self.m_button3 = wx.Button( sbSizer31.GetStaticBox(), wx.ID_ANY, u"确定", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer31.Add( self.m_button3, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		gbSizer1.Add( sbSizer31, wx.GBPosition( 0, 2 ), wx.GBSpan( 2, 1 ), wx.EXPAND, 5 )
		bSizer2 = wx.BoxSizer( wx.VERTICAL )
		self.m_button4 = wx.Button( self, wx.ID_ANY, u"训练", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.m_button4, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_button5 = wx.Button( self, wx.ID_ANY, u"退出", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.m_button5, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		gbSizer1.Add( bSizer2, wx.GBPosition( 2, 2 ), wx.GBSpan( 1, 1 ), wx.EXPAND, 5 )
		sbSizer4 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"训练结果" ), wx.VERTICAL )
		gSizer1 = wx.GridSizer( 0, 2, 0, 0 )
		self.m_bitmap4 = wx.StaticBitmap( sbSizer4.GetStaticBox(), wx.ID_ANY, wx.Bitmap( u"3.jpg", wx.BITMAP_TYPE_ANY ), wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_bitmap4.SetMaxSize( wx.Size( 270,195 ) )
		gSizer1.Add( self.m_bitmap4, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		bSizer3 = wx.BoxSizer( wx.VERTICAL )
		self.m_staticText12 = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, u"训练结果：", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText12.Wrap( -1 )
		bSizer3.Add( self.m_staticText12, 0, wx.ALL, 5 )
		self.m_grid3 = wx.grid.Grid( sbSizer4.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_grid3.CreateGrid( 3, 1 )
		self.m_grid3.EnableEditing( False )
		self.m_grid3.EnableGridLines( True )
		self.m_grid3.EnableDragGridSize( False )
		self.m_grid3.SetMargins( 0, 0 )
		self.m_grid3.EnableDragColMove( False )
		self.m_grid3.EnableDragColSize( True )
		self.m_grid3.SetColLabelValue( 0, u"RMSE" )
		self.m_grid3.SetColLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )
		self.m_grid3.EnableDragRowSize( True )
		self.m_grid3.SetRowLabelValue( 0, u"Y1" )
		self.m_grid3.SetRowLabelValue( 1, u"Y2" )
		self.m_grid3.SetRowLabelValue( 2, u"Y3" )
		self.m_grid3.SetRowLabelValue( 3, u"Y4" )
		self.m_grid3.SetRowLabelAlignment( wx.ALIGN_CENTER, wx.ALIGN_CENTER )
		self.m_grid3.SetDefaultCellAlignment( wx.ALIGN_LEFT, wx.ALIGN_TOP )
		bSizer3.Add( self.m_grid3, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		self.m_staticText15 = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, u"保存预测数据", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText15.Wrap( -1 )
		bSizer3.Add( self.m_staticText15, 0, wx.ALL, 5 )
		self.m_staticText16 = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, u"预测数据已被保存到：...中", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText16.Wrap( -1 )
		bSizer3.Add( self.m_staticText16, 0, wx.ALIGN_CENTER|wx.ALL, 5 )
		gSizer1.Add( bSizer3, 1, wx.EXPAND, 5 )
		sbSizer4.Add( gSizer1, 1, wx.EXPAND, 5 )
		gbSizer1.Add( sbSizer4, wx.GBPosition( 1, 0 ), wx.GBSpan( 2, 2 ), wx.EXPAND, 5 )
		self.SetSizer( gbSizer1 )
		self.Layout()
		self.Centre( wx.BOTH )
		self.m_filePicker1.Bind( wx.EVT_FILEPICKER_CHANGED, self.LOAD_DATA )
		self.m_button1.Bind( wx.EVT_BUTTON, self.DATA_CONFIRM )
		self.m_button2.Bind( wx.EVT_BUTTON, self.MODULE_CONFIRM )
		self.m_button3.Bind( wx.EVT_BUTTON, self.TRAIN_SET_CONFIRM )
		self.m_button4.Bind( wx.EVT_BUTTON, self.TRAIN_START )
		self.m_button5.Bind( wx.EVT_BUTTON, self.MAIN_QUIT )
		self.LSTM_layers.SetValue(u'2')
		self.LSTM_cells.SetValue(u'5')
		self.Y_.SetValue(u'1')
		self.m_grid2.SetCellValue(0,0,u'0.6')
		self.m_grid2.SetCellValue(1,0,u'0.2')
		self.m_grid2.SetCellValue(2,0,u'0.2')
		self.Learing_rate.SetValue(u'0.005')
		self.m_textCtrl5.SetValue(u'10')
		self.m_textCtrl6.SetValue(u'10')
		self.m_checkBox2.Enable(False)
		self.m_grid2.Enable(False)
		self.LSTM_cells.Enable(False)
		self.LSTM_layers.Enable(False)
		self.m_button2.Enable(False)
		self.m_radioBox4.Enable(False)
		self.m_radioBox5.Enable(False)
		self.Learing_rate.Enable(False)
		self.m_textCtrl5.Enable(False)
		self.m_textCtrl6.Enable(False)
		self.m_button3.Enable(False)
		self.m_button4.Enable(False)
		self.m_staticText16.Enable(False)
	def __del__( self ):
		pass	
	def LOAD_DATA( self, event ):#加载数据检查
		global data_path
		data_path=self.m_filePicker1.GetPath()
		print(data_path)
		try:
			input_data=np.loadtxt(str(data_path),dtype=np.float16,delimiter=",",skiprows=1)
			self.m_grid1.SetCellValue(0,0,str(input_data.shape[0]))
			self.m_grid1.SetCellValue(1,0,(str(input_data.shape[1]-1)))
			self.m_button1.Enable(True)
		except:
			input_warn=wx.MessageDialog(None,"请检查导入的数据表！","输入数据错误！",wx.YES_DEFAULT|wx.ICON_QUESTION)
			if input_warn.ShowModal()==wx.ID_YES:
				input_warn.Destroy()
		finally:
			event.Skip()
	def DATA_CONFIRM( self, event ):#训练集设置确认按钮
		global Y_cols
		Y_cols=int(self.Y_.GetValue())
		if Y_cols>=int(self.m_grid1.GetCellValue(1,0)):
			Y_warn=wx.MessageDialog(None,"因变量超出范围！","因变量设置错误！",wx.YES_DEFAULT|wx.ICON_QUESTION)
			if Y_warn.ShowModal()==wx.ID_YES:
				Y_warn.Destroy()
			return None
		DOUBLE_ENABLE=self.m_checkBox3.GetValue()
		DATA_MAKE(DOUBLE_ENABLE,data_path)
		self.m_filePicker1.Enable(False)
		self.Y_.Enable(False)
		self.m_grid1.Enable(False)
		self.m_checkBox3.Enable(False)
		self.m_button1.Enable(False)
		self.m_checkBox2.Enable(True)
		self.m_grid2.Enable(True)
		self.LSTM_cells.Enable(True)
		self.LSTM_layers.Enable(True)
		self.m_button2.Enable(True)
		event.Skip()
	def MODULE_CONFIRM( self, event ):#模型检查
		Dense_use=self.m_checkBox2.GetValue()
		layers=int(self.LSTM_layers.GetValue())
		cells=int(self.LSTM_cells.GetValue())
		train_vol=float(self.m_grid2.GetCellValue(0,0))
		val_vol=float(self.m_grid2.GetCellValue(1,0))
		test_vol=float(self.m_grid2.GetCellValue(2,0))
		DATA_RESHAPE(scaled,train_vol,val_vol,test_vol)
		LAYERS(model,Dense_use,layers,cells)
		self.m_checkBox2.Enable(False)
		self.m_grid2.Enable(False)
		self.LSTM_cells.Enable(False)
		self.LSTM_layers.Enable(False)
		self.m_button2.Enable(False)
		self.m_radioBox4.Enable(True)
		self.m_radioBox5.Enable(True)
		self.Learing_rate.Enable(True)
		self.m_textCtrl5.Enable(True)
		self.m_textCtrl6.Enable(True)
		self.m_button3.Enable(True)
		event.Skip()
	def TRAIN_SET_CONFIRM(self,event):
		Compiler_order=self.m_radioBox4.GetSelection()
		global lr,epochs,batch_size
		try:
			lr=float(self.Learing_rate.GetValue())
			epochs=int(self.m_textCtrl5.GetValue())
			batch_size=int(self.m_textCtrl6.GetValue())
		except:
			warn2=wx.MessageDialog(None,"请检查训练设置!","训练设置错误",wx.YES_DEFAULT|wx.ICON_QUESTION)
			if warn2.ShowModal() == wx.ID_YES:
				warn2.Destroy()
			return None
		loss_function_order=self.m_radioBox5.GetSelection()
		if Compiler_order==0:
			Compiler='SGD'
		elif Compiler_order==1:
			Compiler='rmsprop'
		else:
			Compiler='adam'
		if loss_function_order==0:
			loss_function='mse'
		elif loss_function_order==1:
			loss_function='msle'
		else:
			loss_function='mae'
		MODEL_SET(Compiler,loss_function)
		self.m_radioBox4.Enable(False)
		self.m_radioBox5.Enable(False)
		self.Learing_rate.Enable(False)
		self.m_textCtrl5.Enable(False)
		self.m_textCtrl6.Enable(False)
		self.m_button3.Enable(False)
		self.m_button4.Enable(True)
		event.Skip()
	def TRAIN_START( self, event ):
		MODEL_TRAIN(lr,epochs,batch_size)
		PIC=wx.Image("./process.png",wx.BITMAP_TYPE_PNG)
		self.m_bitmap4.SetBitmap(wx.Bitmap(PIC.Rescale(270,195).ConvertToBitmap()))
		RMSE_=MODEL_PREDICT()
		for i in range(Y_cols):
			self.m_grid3.SetCellValue(i,0,str(RMSE_[i]))
		self.m_staticText16.Enable(True)
		self.m_staticText16.SetLabelText("预测数据已被保存到./output.csv中")
		event.Skip()
	def MAIN_QUIT( self, event ):
		self.Destroy()
		event.Skip()
