import backtrader as bt
import numpy as np
from backtrader import talib
from functools import reduce
from scipy.signal import argrelextrema
from datetime import datetime
import sys, inspect



startcash = 1000
datafile = 'data.csv'
data = bt.feeds.GenericCSVData(dataname=datafile,
                               # fromdate=datetime(2021, 8, 1),
                               openinterest=-1,
                               timeframe=bt.TimeFrame.Minutes,
                               dtformat=('%Y-%m-%d %H:%M:%S'))
candles_analyzed = 0


def line():
    "Gives the script lign number when an error is raised"
    return inspect.currentframe().f_back.f_lineno


def deepgetattr(obj: object, attr: str, val_default: float = 0.0) -> float:
    """Recurses through an attribute chain to get the ultimate value."""
    try:
        res = reduce(getattr, attr.split('.'), obj)
        return res
    except KeyError:
        return val_default


class CommInfoFractional(bt.CommissionInfo):
    params = dict(commission=0.00018)  # 0.018% commission

    def getsize(self, price, cash):
        '''Returns fractional size for cash operation @price'''
        return (cash / price)


class St(bt.Strategy):
    params = (
        ('risk_reward', 3),  # multiple
        ('radius', 1),

        ('order_log', True)
    )

    _riskreward_list = []

    def __init__(self):
        self.EMA20 = talib.EMA(self.data.close, timeperiod=20)
        self.EMA8 = talib.EMA(self.data.close, timeperiod=8)
        self.ATR = talib.ATR(self.data.high, self.data.low, self.data.close, timeperiod=13)
        self.crossover = bt.ind.CrossOver(self.data.close, self.EMA20)
        self.riskreward = None

    def riskrewardfun(self, sl, entry, exitt):
        return abs((exitt - entry)/(entry - sl))

    def log(self, txt):
        ''' Logging function for this strategy'''
        dt = self.datas[0].datetime.datetime(0)
        if (self.params.order_log):
            print('%s, %s' % (dt, txt))

    def stoploss(self, order_type, entry_price):
        highest_high_13days = max([self.data.high[-i] for i in range(1, 2)])
        lowest_low_13days = min([self.data.low[-i] for i in range(1, 2)])
        list_percent = [self.ATR[-i] / self.data.close[-i] for i in range(1, 52)]
        average_percent = sum(list_percent) / len(list_percent)

        if (order_type == "BUY"):
            minimum_stoploss = entry_price * (1 - average_percent * 1.05)
            dist = entry_price - lowest_low_13days
            if dist > 2 * self.ATR[0]:
                stop_loss_ATR = entry_price - self.ATR[0]
            else:
                stop_loss_ATR = lowest_low_13days - 0.5 * self.ATR[0]
            return min(stop_loss_ATR, minimum_stoploss)

        if (order_type == "SELL"):
            minimum_stoploss = entry_price * (1 + average_percent * 1.05)
            dist = highest_high_13days - entry_price
            if dist > 2 * self.ATR[0]:
                stop_loss_ATR = entry_price + self.ATR[0]
            else:
                stop_loss_ATR = highest_high_13days + 0.5 * self.ATR[0]
            return max(stop_loss_ATR, minimum_stoploss)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        if order.status in [order.Completed]:
            if self.position.size == 0 or 'name' in order.info:
                self._riskreward_list.append(self.riskreward(order.executed.price))
            if 'name' in order.info:
                self.log('POSITION CLOSED, %.4f, %.4f' % (order.executed.price, order.executed.size))
            elif self.position.size == 0:
                self.log('POSITION CLOSED, %.4f, %.4f' % (order.executed.price, order.executed.size))
                for i in self.broker.orders:
                    if i.status < 4:
                        self.cancel(i)
            else:
                if order.isbuy():
                    # set stoploss order
                    entry_price = order.executed.price
                    sl = self.stoploss('BUY', entry_price)
                    sl_order = self.close(exectype=bt.Order.Stop, price=sl)
                    sl_order.addinfo(name="Stop Loss")

                    # set target order
                    target = order.executed.price + self.params.risk_reward * (order.executed.price - sl)
                    target_order = self.close(exectype=bt.Order.Limit, price=target, oco=sl_order)
                    target_order.addinfo(name="Profit")
                    self.log('BUY EXECUTED, %.4f, %.4f, %.4f' % (order.executed.price, sl, target))
                elif order.issell():
                    # set stoploss order
                    entry_price = order.executed.price
                    sl = self.stoploss('SELL', entry_price)
                    sl_order = self.close(exectype=bt.Order.Stop, price=sl)
                    sl_order.addinfo(name="Stop Loss")

                    # set target order
                    target = order.executed.price - self.params.risk_reward * (sl - order.executed.price)
                    target_order = self.close(exectype=bt.Order.Limit, price=target, oco=sl_order)
                    target_order.addinfo(name="Profit")
                    self.log('SELL EXECUTED, %.4f, %.4f, %.4f' % (order.executed.price, sl, target))

            self.bar_executed = len(self)

        elif order.status in [order.Margin, order.Rejected]:
            self.log('Order Margin/Rejected')

    def next(self):
        if self.position.size == 0:
            order_type = None
            if self.EMA8[0] > self.EMA20[0] and self.EMA8[-1] < self.EMA20[-1]:
                self.order_target_percent(target=0.02)
                order_type = 'BUY'
            ### Short Signal
            if self.EMA8[0] < self.EMA20[0] and self.EMA8[-1] > self.EMA20[-1]:
                self.order_target_percent(target=-0.02)
                order_type = 'SELL'
            if order_type == 'BUY':
                self.riskreward = RiskReward(price_entry=self.data.close[0], price_stoploss=self.stoploss('BUY', self.data.close[0]))
            elif order_type == 'SELL':
                self.riskreward = RiskReward(price_entry=self.data.close[0], price_stoploss=self.stoploss('SELL', self.data.close[0]))
        else:
            if (self.position.size > 0) and (self.crossover < 0):
                self.close()
            if (self.position.size < 0) and (self.crossover > 0):
                self.close()
            return

    def stop(self):
        pnl = round(self.broker.getvalue() - startcash, 2)
        self.value = round(self.broker.get_value(), 2)
        global candles_analyzed
        candles_analyzed = len(self.data.open)


def printTradeAnalysis(analyzer):
    '''
	Function to print the Technical Analysis results in a nice format.
	'''
    # Get the results we are interested in
    try:
        longs_won = round(analyzer.long.won, 2)
    except:
        longs_won = 0

    try:
        total_longs = round(analyzer.long.total, 2)
    except:
        total_longs = 0

    try:
        shorts_won = round(analyzer.short.won, 2)
    except:
        shorts_won = 0

    try:
        total_shorts = round(analyzer.short.total, 2)
    except:
        total_shorts = 0

    try:
        long_win_rate = round(longs_won / total_longs * 100, 2)
    except:
        long_win_rate = 0

    try:
        short_win_rate = round(shorts_won / total_shorts * 100, 2)
    except:
        short_win_rate = 0

    try:
        total_closed = round(analyzer.total.closed, 2)
    except:
        total_closed = 0

    try:
        total_won = round(analyzer.won.total, 2)
    except:
        total_won = 0

    try:
        win_streak = round(analyzer.streak.won.longest, 2)
    except:
        win_streak = 0

    try:
        lose_streak = round(analyzer.streak.lost.longest, 2)
    except:
        lose_streak = 0

    try:
        pnl_net = round(analyzer.pnl.net.total, 2)
    except:
        pnl_net = 0

    try:
        win_rate = round((total_won / total_closed) * 100, 2)
    except:
        win_rate = 0

    try:
        total_return = 100 * pnl_net / startcash
    except:
        total_return = 0

    try:
        frequncy = analyzer.total.total / candles_analyzed
    except:
        frequncy = 0

    try:
        expectancy = (win_rate / 100) * (np.mean(deepgetattr(St, '_riskreward_list')) - 0.07) - (1 - (win_rate / 100)) - 0.12
        freq_expectancy = round(total_closed * expectancy, 3)
    except:
        expectancy = 0
        freq_expectancy = 0

    # Designate the rows
    h1 = ['Win rate', 'Longs Won', 'Long Win Rate', 'Shorts Won', 'Short Win Rate']
    r1 = [win_rate, '{}/{}'.format(longs_won, total_longs), long_win_rate, '{}/{}'.format(shorts_won, total_shorts),
          short_win_rate]
    h2 = ['Total Trades', 'Win Streak', 'Lose Streak', 'freq_expectancy', '% Total Return']
    r2 = [total_closed, win_streak, lose_streak, freq_expectancy, total_return]
    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (len(h1))
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format(*row))


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(startcash)
    cerebro.adddata(data)
    cerebro.addstrategy(St)
    cerebro.broker.addcommissioninfo(CommInfoFractional())
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    strategies = cerebro.run()

    # print the analyzers
    printTradeAnalysis(strategies[0].analyzers.ta.get_analysis())
