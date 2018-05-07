# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Jun 17 2015)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import sccn_test_app

###########################################################################
## Class zongti
###########################################################################

class zongti(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"change detection", pos=wx.DefaultPosition,
                          size=wx.Size(600, 300), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHintsSz(wx.Size(600, 300), wx.Size(600, 300))

        zuida = wx.GridBagSizer(0, 0)
        zuida.SetFlexibleDirection(wx.BOTH)
        zuida.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        zuo = wx.GridBagSizer(0, 0)
        zuo.SetFlexibleDirection(wx.BOTH)
        zuo.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        wenjian = wx.GridBagSizer(0, 0)
        wenjian.SetFlexibleDirection(wx.BOTH)
        wenjian.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.OPT = wx.StaticText(self, wx.ID_ANY, u"optical image", wx.DefaultPosition, wx.DefaultSize, 0)
        self.OPT.Wrap(-1)
        wenjian.Add(self.OPT, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.OPT_F = wx.FilePickerCtrl(self, wx.ID_ANY, wx.EmptyString, u"Select a file", u"*.*", wx.DefaultPosition,
                                       wx.DefaultSize, wx.FLP_DEFAULT_STYLE)
        wenjian.Add(self.OPT_F, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.SAR = wx.StaticText(self, wx.ID_ANY, u"sar image    ", wx.DefaultPosition, wx.DefaultSize, 0)
        self.SAR.Wrap(-1)
        wenjian.Add(self.SAR, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.SAR_F = wx.FilePickerCtrl(self, wx.ID_ANY, wx.EmptyString, u"Select a file", u"*.*", wx.DefaultPosition,
                                       wx.DefaultSize, wx.FLP_DEFAULT_STYLE)
        wenjian.Add(self.SAR_F, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.SAVE = wx.StaticText(self, wx.ID_ANY, u"save", wx.DefaultPosition, wx.DefaultSize, 0)
        self.SAVE.Wrap(-1)
        wenjian.Add(self.SAVE, wx.GBPosition(2, 0), wx.GBSpan(1, 1), wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.SAVE = wx.DirPickerCtrl(self, wx.ID_ANY, wx.EmptyString, u"Select a folder to save", wx.DefaultPosition,
                                     wx.DefaultSize, wx.DIRP_DEFAULT_STYLE)
        wenjian.Add(self.SAVE, wx.GBPosition(2, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        zuo.Add(wenjian, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.EXPAND, 5)

        canshu = wx.GridBagSizer(0, 0)
        canshu.SetFlexibleDirection(wx.BOTH)
        canshu.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        pretrain = wx.GridBagSizer(0, 0)
        pretrain.SetFlexibleDirection(wx.BOTH)
        pretrain.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.NOISE = wx.StaticText(self, wx.ID_ANY, u"noise std", wx.DefaultPosition, wx.DefaultSize, 0)
        self.NOISE.Wrap(-1)
        pretrain.Add(self.NOISE, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.PRETRAIN = wx.CheckBox(self, wx.ID_ANY, u"pretrain", wx.DefaultPosition, wx.DefaultSize, 0)
        self.PRETRAIN.SetValue(True)
        pretrain.Add(self.PRETRAIN, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.NOISE_T = wx.TextCtrl(self, wx.ID_ANY, u"0.02", wx.DefaultPosition, wx.Size(85, -1), 0)
        pretrain.Add(self.NOISE_T, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.TIMES = wx.StaticText(self, wx.ID_ANY, u"times", wx.DefaultPosition, wx.DefaultSize, 0)
        self.TIMES.Wrap(-1)
        pretrain.Add(self.TIMES, wx.GBPosition(2, 0), wx.GBSpan(1, 1), wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.STEP = wx.StaticText(self, wx.ID_ANY, u"step", wx.DefaultPosition, wx.DefaultSize, 0)
        self.STEP.Wrap(-1)
        pretrain.Add(self.STEP, wx.GBPosition(3, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.TIMES_T = wx.TextCtrl(self, wx.ID_ANY, u"3000", wx.DefaultPosition, wx.Size(84, -1), 0)
        pretrain.Add(self.TIMES_T, wx.GBPosition(2, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.STEP_T = wx.TextCtrl(self, wx.ID_ANY, u"0.01", wx.DefaultPosition, wx.Size(85, -1), 0)
        pretrain.Add(self.STEP_T, wx.GBPosition(3, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        canshu.Add(pretrain, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.EXPAND, 5)

        train = wx.GridBagSizer(0, 0)
        train.SetFlexibleDirection(wx.BOTH)
        train.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.TRAIN = wx.StaticText(self, wx.ID_ANY, u"train", wx.DefaultPosition, wx.DefaultSize, 0)
        self.TRAIN.Wrap(-1)
        train.Add(self.TRAIN, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.LAMBDA = wx.StaticText(self, wx.ID_ANY, u"lambda", wx.DefaultPosition, wx.DefaultSize, 0)
        self.LAMBDA.Wrap(-1)
        train.Add(self.LAMBDA, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.LAMBDA_T = wx.TextCtrl(self, wx.ID_ANY, u"0.1", wx.DefaultPosition, wx.Size(85, -1), 0)
        train.Add(self.LAMBDA_T, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.TTIMES = wx.StaticText(self, wx.ID_ANY, u"times", wx.DefaultPosition, wx.DefaultSize, 0)
        self.TTIMES.Wrap(-1)
        train.Add(self.TTIMES, wx.GBPosition(2, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.STEP = wx.StaticText(self, wx.ID_ANY, u"step", wx.DefaultPosition, wx.DefaultSize, 0)
        self.STEP.Wrap(-1)
        train.Add(self.STEP, wx.GBPosition(3, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.TSTEP = wx.TextCtrl(self, wx.ID_ANY, u"0.01", wx.DefaultPosition, wx.Size(85, -1), 0)
        train.Add(self.TSTEP, wx.GBPosition(3, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        self.TTIMES_T = wx.TextCtrl(self, wx.ID_ANY, u"150", wx.DefaultPosition, wx.Size(85, -1), 0)
        train.Add(self.TTIMES_T, wx.GBPosition(2, 1), wx.GBSpan(1, 1), wx.ALL, 5)

        canshu.Add(train, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.EXPAND, 5)

        zuo.Add(canshu, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.EXPAND, 5)

        zuida.Add(zuo, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.EXPAND, 5)

        you = wx.GridBagSizer(0, 0)
        you.SetFlexibleDirection(wx.BOTH)
        you.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

        self.START = wx.Button(self, wx.ID_ANY, u"start", wx.DefaultPosition, wx.Size(130, -1), 0)
        you.Add(self.START, wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        self.info = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size(250, 200), wx.TE_MULTILINE)
        you.Add(self.info, wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL, 5)

        zuida.Add(you, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.EXPAND, 5)

        self.SetSizer(zuida)
        self.Layout()

        self.Centre(wx.BOTH)

        # Connect Events
        self.OPT_F.Bind(wx.EVT_FILEPICKER_CHANGED, self.opt_change)
        self.SAR_F.Bind(wx.EVT_FILEPICKER_CHANGED, self.sar_change)
        self.SAVE.Bind(wx.EVT_DIRPICKER_CHANGED, self.save_change)
        self.PRETRAIN.Bind(wx.EVT_CHECKBOX, self.pretrain_b)
        self.START.Bind(wx.EVT_BUTTON, self.start)

    def __del__(self):
        pass

    # Virtual event handlers, overide them in your derived class
    def opt_change(self, event):

        event.Skip()

    def sar_change(self, event):

        event.Skip()

    def save_change(self, event):


        event.Skip()

    def pretrain_b(self, event):
        if self.PRETRAIN.GetValue():
            self.info.AppendText('-->Pretrain\n')
        event.Skip()

    def start(self, event):
        self.info.Clear()
        opt_path = self.OPT_F.GetPath()
        sar_path = self.SAR_F.GetPath()
        save_path = self.SAVE.GetPath()
        noise_std = float(self.NOISE_T.GetValue())
        p_times = int(self.TIMES_T.GetValue())
        p_step = float(self.STEP_T.GetValue())
        lam = float(self.LAMBDA_T.GetValue())
        t_times = int(self.TTIMES_T.GetValue())
        t_step = float(self.TSTEP.GetValue())
        self.info.AppendText('Ready!\n')
        self.info.AppendText('-->opt:'+opt_path)
        self.info.AppendText('\n-->sar:'+sar_path)
        self.info.AppendText('\n-->pretrain:' )
        self.info.AppendText('\n   noise std: ' + str(noise_std))
        self.info.AppendText('\n   times: ' + str(p_times))
        self.info.AppendText('\n   step: ' + str(p_step))
        self.info.AppendText('\n-->train:' )
        self.info.AppendText('\n   λ: ' + str(lam))
        self.info.AppendText('\n   times: ' + str(t_times))
        self.info.AppendText('\n   step:' + str(t_step))
        self.info.AppendText('\n-----------' )
        sccn_test_app.SCCN(opt_path,sar_path,save_path,noise_std,p_times,p_step,lam,t_times,t_step,self.info)
        event.Skip()


app=wx.App()
m = zongti(None)
m.Show()
app.MainLoop()
