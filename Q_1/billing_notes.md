# Handover notes from the billing vendor (RetailEdge POS)

Whoever picks this up next -- read this before you touch the exports. I am
leaving the account and nobody here has written this down before.

## The nightly export

Every store's till server writes one file per **trading day** into the shared
folder at 02:30 local time. The file name is the contract, not the contents:

    SALES_<store_id>_<YYYYMMDD>.<csv|parquet>

The date in the file name is the **business date**. Bills punched between
midnight and roughly 01:30 belong to the business day that just ended, so you
will see timestamps inside a file that are on the *following* calendar date.
Trust the file name. We have never fixed this and finance has never minded.

## Re-sends

If a store phones in a problem, the ops desk re-triggers the export. The
re-sent file lands next to the original with a suffix:

    SALES_S03_20241014.csv          <- original
    SALES_S03_20241014__R1.csv      <- re-send
    SALES_S03_20241014__R2.csv      <- second re-send

Two things to know.

1. Most re-sends are a complete replacement and are byte-identical to the
   original.
2. Some are **not complete**. If the till was mid-roll when we re-triggered,
   the re-send only contains the bills that had been committed at that moment.
   It is still a valid file, it just has fewer bills in it. I know of at least
   a few of these in November.

So "take the newest file for each store-day" is wrong. The safe unit is the
**line**, identified by `(bill_no, line_no)`. `bill_no` already contains the
store and the business date.

## Line types

One bill produces several rows. `line_type` tells you what a row is.

| line_type  | what it is                                              | counts as revenue? |
|------------|---------------------------------------------------------|--------------------|
| `SALE`     | one item on the bill, positive qty                        | yes                |
| `RETURN`   | one item on a return bill, negative qty                   | yes (it subtracts) |
| `DISCOUNT` | bill-level discount, qty 1, negative `unit_price`         | yes (it subtracts) |
| `VOID`     | mirror of an item line on a cancelled bill, negated qty   | yes (it cancels)   |
| `TAX`      | GST for the whole bill, qty 1                             | **no**             |
| `TENDER`   | what the customer actually paid: items - discount + tax   | **no**             |

The one that catches everybody: **`TENDER` is the bill total, written as
another row in the same file.** If you `SUM(qty * unit_price)` over every row
you will count each bill roughly twice, and you will count the GST as well.
The finance team does not count GST as revenue.

Cancelled bills are handled by writing the item lines a second time with
`line_type = 'VOID'` and the quantity negated. So a cancelled bill nets to
zero if you keep both, and inflates your revenue if you filter out only the
`VOID` rows and keep the `SALE` rows. Either keep both, or drop every line of
any `bill_no` that has a `VOID` row. Do not do half of it.

## Product codes

This is the ugly one. In June 2024 merchandising ran a catalogue clean-up and
**reissued about two dozen product codes that had been retired** to completely
different products -- in some cases in a different category. The code on the
bill therefore does not identify the product on its own. You need the code
**and the date of the sale**, checked against `products.valid_from` /
`products.valid_to` in PostgreSQL.

If you join on `product_code` alone you will get two master rows back for
those codes, every sale line of theirs will be duplicated, and half of them
will be filed under the wrong category.

## Prices

`unit_price` on the export is the price the till actually printed. It is
usually right. It is not always right -- a till that has not synced picks up
the previous price list, and we see this on roughly one line in seventy. The
authoritative price list is `price_revisions` in PostgreSQL, keyed by
`product_sk` with `effective_from` / `effective_to`. If somebody asks what
something sold for in March, answer from `price_revisions` as of March, not
from today's row and not from the printed price.

## File dialects

The three till software generations never got unified.

* Stores S01-S05: comma separated, ISO-8601 timestamps
  (`2024-10-14T13:45:02`), headers `bill_no,line_no,product_code,qty,unit_price,line_type,ts`.
* Stores S06-S09: **semicolon** separated, `dd-mm-yyyy HH:MM:SS` timestamps,
  headers `bill_no;line_no;item_code;quantity;rate;type;txn_time`.
* Stores S10-S12: comma separated but the file starts with a **UTF-8 BOM**,
  timestamps are **epoch seconds (UTC)**, and the columns are in a different
  order.

Some days are written as Parquet instead of CSV. That was an experiment by the
previous vendor and it was never rolled back or rolled forward.

## Known gap

Pune (S07) lost its till server for three days in July 2024. Those exports do
not exist and will never exist. Finance has those numbers because the store
phoned them in.
