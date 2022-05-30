---
title: Trade Lifecycle
date: 2022-05-21
description: Udemy course notes
category: summary
type: notes
---

Series of steps through which any trade passes. Objective is to have the trade settle by the due date.

Front office:

- trade execution (buy/sell from counterparty)
- trade capture (trade skeleton should be captured)

Both front and back office need reference data

- security info
- counterparty info

Operations:

- trade capture
- trade enrichment
- trade confirmation (ensure details captured by party/counterparty are agreed upon)

- settlement instructions (command issued to depository/custodian that will actually authorize the trade settlement)
- pre-settlement statuses (depository makes sure settlement instructions match)
- failed settlement (5-10% of trades fail due to non matching instructions)
- settlement
- reflecting settlment
- reconciliation (update local book of record to match depository)

### What is a trade?

Legal agreement to buy or sell goods. One party pays cash in exchange for goods, other party receives cash in exchange for goods.

Parties can be investor & broker, broker & exchange, broker & trader, trader & trader

### Trade Execution

Historically, between traders on a trading floor where market makers advertise their prices on "pitches"

Currently, automatic execution via electronic exchanges (order-driven markets) or bid/offer prices are quoted to attract investors (quote-driven markets).

trader uses current trading position + average price of current trading position to determine whether to buy/sell at profit/loss.

Investment manager makes an investment decision and communicates an "order" to executing broker. Broker records detail of the order - client account, buy/sell, quantity, security, desired price.

When the broker receives the order, it acts as agent (broker attempts to find a third party willing to trade on terms of order - investment bank makes commission _only_) or principal (trader trades off investment bank trading book). The broker becomes the counterparty to the trade (broker is investors counterparty and investor is broker's counterparty).

If the trade is made by a fund (ie mutual fund provider), the block level trade needs to be allocated to its underlying funds (splits).
