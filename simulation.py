# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
from datetime import datetime, timedelta


os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('./Data_create')
df_day_f = pd.read_csv('df_day_forward.csv', index_col=0)
df_day_f['Date'] = pd.to_datetime(df_day_f['Date'])
#df_day_f = df_day_f[df_day_f['Date'] >= pd.to_datetime('2012-04-23')].reset_index(drop=True)

df_5min_f = pd.read_csv('df_5min_forward.csv', index_col=0)
df_5min_f['Date'] = pd.to_datetime(df_5min_f['Date'])
#df_5min_f = df_5min_f[df_5min_f['Date'] >= pd.to_datetime('2012-04-23')].reset_index(drop=True)


os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('df_inout.csv', index_col=0)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'] >= pd.to_datetime('2001-01-04')].reset_index(drop=True)
df = df[df['Date'] <= pd.to_datetime('2023-05-22')].reset_index(drop=True)
df = df.drop(["Deviation","Increase Rate","Kyousi","Day","funds","total assets"], axis=1, inplace=False)



#df = df.dropna().reset_index(drop=True)

df['Funds'] = np.nan
df['Bull'] = np.nan
df['Bear'] = np.nan

df.to_csv('memo.csv')


#予測期間：2012/4/23～2023/5/22

#日経225先物シミュレーション
class Nikkei_Futures():
    def __init__(self, pocket_fund, df, df_day_f, df_5min_f):
        self.fund = pocket_fund #所持金
        self.df = df #日経予測データ
        self.df_day_f = df_day_f #日経225先物 日足終値
        self.df_5min_f = df_5min_f #日経225先物 5分足終値

        self.tax = 0.1 #消費税[]
        self.commision = 300 #売買手数料[円/枚]
        self.loss_cut = 300 #損切り値[円]
        self.profit = 30 #利食い値[円]
        self.alpha = 0.8 #α
        self.margin = 132 * 10000 #証拠金[円]

        self.bull_account = 0 #買い玉(上限200)
        self.bear_account = 0 #売り玉(上限200)
        self.bull_arr = [] #買い玉枚数、価格配列
        self.bear_arr = [] #売り玉枚数、価格配列
        self.day = 0
        self.num = 0

        self.profit_count = 0
        self.losscut_count = 0
        self.profit_arr = []
        self.losscut_arr = []

    
    def __call__(self):
        #self.df['Funds'][day] = int(self.fund / 10000)
        #self.df['Bull'][day] = self.bull_account
        #self.df['Bear'][day] = self.bear_account
        self.df.loc[self.df['Date'] == self.day, ['Funds']] = int(self.fund / 10000)
        self.df.loc[self.df['Date'] == self.day, ['Bull']] = self.bull_account
        self.df.loc[self.df['Date'] == self.day, ['Bear']] = self.bear_account
    
    #先行買い
    def pre_buy(self):
        trade_money = self.fund * 0.2 - (self.bull_account + self. bear_account) * self.margin #売買資金
        trade_num = int(trade_money / self.margin) #売買数
        if trade_num > 0:
            commision = trade_num * 275 #手数料[円/枚]
            self.fund = self.fund - commision
            #買い玉が上限200個を超えるとき
            if self.bull_account + trade_num > 200:
                trade_num = 400 - self.bull_account #上限200枚
                pass
            if trade_num > 0:
                self.bull_account = self.bull_account + trade_num #買い玉更新
                #[取引枚数、取引価格、0]
                self.bull_arr.append([trade_num, self.df_day_f.loc[self.df_day_f['Date'] == self.day, "Day Close"].values[0], [0]])


    #先行売り
    def pre_sell(self):
        trade_money = self.fund * 0.2 - (self.bull_account + self. bear_account) * self.margin #売買資金
        trade_num = int(trade_money / self.margin) #売買数
        if trade_num > 0:
            commision = trade_num * 275 #手数料[円/枚]
            self.fund = self.fund - commision
            #売り玉が上限200個を超えるとき
            if self.bear_account + trade_num > 200:
                trade_num = 200 - self.bear_account #上限200枚
                pass
            if trade_num > 0:
                self.bear_account = self.bear_account + trade_num #売り玉更新
                #[取引枚数、取引価格、0]
                self.bear_arr.append([trade_num, self.df_day_f.loc[self.df_day_f['Date'] == self.day, "Day Close"].values[0], [0]])





    #反対買い
    def counter_buy(self, diff):
        # self.bear_arr の self.num 番目の売り玉を売る
        trade_num = self.bear_arr[self.num][0]
        commision = trade_num * 275 #手数料[円/枚]
        self.fund = self.fund + diff * 1000 * trade_num - commision
        self.bear_arr.pop(self.num)
        self.bear_account = self.bear_account - trade_num

    #反対売り
    def counter_sell(self, diff_bull):
        # self.bull_arr の self.num 番目の買い玉を売る
        trade_num = self.bull_arr[self.num][0]
        commision = trade_num * 275 #手数料[円/枚]
        self.fund = self.fund + diff_bull * 1000 * trade_num - commision
        self.bull_arr.pop(self.num)
        self.bull_account = self.bull_account - trade_num
        

    #取引
    def trade(self):
        for day in range(1, len(self.df)):

            self.day = self.df['Date'][day]
            #資金不足の場合
            if self.fund <= 660 * 10000:
                print("資金の20%が証拠金を下回りました。これ以上取引は出来ません。")
                break
            #先物データがない日付の場合
            if self.df_day_f[self.df_day_f['Date']==self.day].empty:
                print("先物データがありません。",self.day)
                bt()
                continue
            #反対売買------------------------------------------------------------------------------------------
            #買い玉を持っているとき
            if self.bull_account!=0:
                section_df = self.df_day_f.loc[(self.df_day_f['Date'] == self.day)|(self.df_day_f['Date'] == self.df['Date'][day-1])].reset_index(drop=True)
                if len(section_df) != 2:
                    section_df = self.df_day_f.loc[(self.df_day_f['Date'] == self.day)|(self.df_day_f['Date'] == self.df['Date'][day-2])].reset_index(drop=True)

                if section_df['Day High'][0] > section_df['Day High'][1]:
                    bull_arr = self.bull_arr #消すとself.bull_arrが変化していくからfor文がうまく回らなくなる
                    for num in range(len(bull_arr)):
                        self.num = num
                        diff_bull = section_df['Day High'][0] - bull_arr[num][1]#正：利益、負：損失
                        #利食い値超え
                        if diff_bull >= self.profit:
                            self.counter_sell(diff_bull)
                            self.profit_count = self.profit_count + 1
                            self.profit_arr.append(diff_bull)
                        #損切りの場合
                        elif diff_bull <= -1 * self.loss_cut:
                            self.counter_sell(diff_bull)
                            self.losscut_count = self.losscut_count + 1
                            self.losscut_arr.append(diff_bull)
                
            #売り玉を持っているとき
            if self.bear_account!=0:
                pass

            '''
            2023_5作
            #買い玉を持っているとき
            if self.bull_account!=0:
                section_df = self.df_5min_f.loc[self.df_5min_f['Date'] == self.day]
                for min in range(len(section_df)):
                    bull_arr = self.bull_arr #消すとself.bull_arrが変化していくからfor文がうまく回らなくなる
                    for num in range(len(bull_arr)):
                        self.num = num
                        diff_bull =  self.df_5min_f['5min Close'][min] - bull_arr[num][1] #正：利益、負：損失
                        #利食い値超え
                        if diff_bull >= self.profit:
                            #最大利益のα％を下回った
                            if diff_bull < max(self.bull_arr[num][2]) * self.alpha:
                                self.counter_sell(diff_bull)
                                self.profit_count = self.profit_count + 1
                                self.profit_arr.append(diff_bull)
                            #最大利益のα％を下回っていない
                            else:
                                self.bull_arr[num][2].append(diff_bull) 
                        #損切りの場合
                        elif diff_bull <= -1 * self.loss_cut:
                            self.counter_sell(diff_bull)
                            print(diff_bull,self.day)
                            self.losscut_count = self.losscut_count + 1
                            self.losscut_arr.append(diff_bull)
                
            #売り玉を持っているとき
            if self.bear_account!=0:
                section_df = self.df_5min_f.loc[self.df_5min_f['Date'] == self.day]
            '''
                


            #先行売買------------------------------------------------------------------------------------------
            if self.df['Predict'][day] == 1:
                #先行売買-------------------------------------------------------
                #下降、上昇
                if self.df['Predict'][day-1] == 0:
                    #資金の20%分買う(切り捨て)
                    self.pre_buy()
                #上昇、上昇、　買い玉がなければ買う
                elif self.bull_account == 0:
                    self.pre_buy()
                

                '''
                #売り玉、反対取引--------------------------------------------------
                if len(self.bear_arr) != 0:
                    #売り玉を持っている場合
                    bear_arr = self.bear_arr
                    for num in range(len(bear_arr)):
                        self.num = num
                        diff = bear_arr[num][1] - self.df['Close Futures'][day] #正：利益、負：損失
                        #損切条件
                        if diff <= -1 * self.loss_cut:
                            self.losscut_count = self.losscut_count + 1
                            self.counter_buy(diff)
                            self.losscut_arr.append(diff)

                        #利食い条件
                        #elif (diff < max(self.bear_arr[num][2]) * self.alpha) & (diff >= self.profit):
                        elif diff < max(self.bear_arr[num][2]) * self.alpha:
                            self.profit_count = self.profit_count + 1
                            self.counter_buy(diff)
                            self.profit_arr.append(diff)
                
                
                #買い玉、利益計算--------------------------------------------------
                if len(self.bull_arr) != 0:
                    #買い玉を持っている場合
                    bull_arr = self.bull_arr
                    for num in range(len(bull_arr)):
                        self.num = num
                        diff = self.df['Close Futures'][day] - bull_arr[num][1] #正：利益、負：損失
                        #利食い条件
                        if diff >= self.profit:
                            self.bull_arr[num][2].append(diff)    
                '''

    
                        
            else:
                #先行取引-------------------------------------------------------
                #上昇、下降
                if self.df['Predict'][day-1] == 1:
                    #資金の20%分空売り(切り捨て)
                    #self.pre_sell()
                    pass
                #下降、下降、　売り玉がなければ空売り
                elif self.bear_account == 0:
                    #self.pre_sell()
                    pass
                


                '''
                #買い玉、反対取引--------------------------------------------------
                if len(self.bull_arr) != 0:
                    #買い玉を持っている場合
                    bull_arr = self.bull_arr
                    for num in range(len(bull_arr)):
                        self.num = num
                        diff = self.df['Close Futures'][day] - bull_arr[num][1] #正：利益、負：損失
                        #損切条件
                        if diff <= -1 *  self.loss_cut:
                            self.losscut_count = self.losscut_count + 1
                            self.counter_sell(diff)
                            self.losscut_arr.append(diff)

                        #利食い条件
                        #elif (diff < max(self.bull_arr[num][2]) * self.alpha) & (diff >= self.profit):
                        elif diff < max(self.bull_arr[num][2]) * self.alpha:
                            self.profit_count = self.profit_count + 1
                            self.counter_sell(diff)
                            self.profit_arr.append(diff)

                #売り玉、利益計算--------------------------------------------------
                if len(self.bear_arr) != 0:
                    #売り玉を持っている場合
                    bear_arr = self.bear_arr
                    for num in range(len(bear_arr)):
                        self.num = num
                        diff = bear_arr[num][1] - self.df['Close Futures'][day] #正：利益、負：損失
                        #利食い条件
                        if diff >= self.profit:
                            self.bear_arr[num][2].append(diff)
                '''


            bt()
            
            
        #シミュレーション終了時に買い玉、売り玉を全て売る-------------------------------------
        '''
        if len(self.bull_arr) != 0:
            #買い玉を持っている場合
            for num in range(len(self.bull_arr)):
                self.num = num
                diff = self.df['Close Futures'][day] - self.bull_arr[num][1] #正：利益、負：損失
                self.counter_sell(diff)
        
        if len(self.bear_arr) != 0:
                #売り玉を持っている場合
                for num in range(len(self.bear_arr)):
                    self.num = num
                    diff = self.bear_arr[num][1] - self.df['Close Futures'][day] #正：利益、負：損失
                    self.counter_buy(diff)
        '''

            
            
        print("利食い回数 : "+ str(self.profit_count)+"  利食い平均[円] : "+str(np.mean(self.profit_arr)))
        print("損切り回数 : "+ str(self.losscut_count)+"  損切り平均[円] : "+str(-np.mean(self.losscut_arr)))
        print("勝率[%] : ",self.profit_count/(self.profit_count+self.losscut_count)*100)
        self.df.to_csv('Simulation_Result.csv')
        



pocket_fund = 2000 * 10000 #所持金[円]

bt = Nikkei_Futures(pocket_fund, df, df_day_f, df_5min_f) #バックテスト
bt.trade()


#プロット-----------------------------------------------------------------------------------------

