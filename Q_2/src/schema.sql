-- ---------------------------------------------------------------------------
-- SetuBid deduplication schema  --  PART (d)
--
-- Everything the lookup in part (c) consults at query time lives here as
-- relational data. Nothing that matters survives only in a Python process:
-- kill the job mid-run, restart it, and the index is still there and still
-- queryable by the application.
--
-- Four groups of tables:
--   1. notice                  the corpus (what the product already owns)
--   2. notice_sketch           the reduced form from part (b)
--   3. lsh_bucket              the retrieval structure from part (c)
--   4. opportunity / alias     stable card identity, because a bookmark taken
--      / merge_decision        today must still resolve next month
-- ---------------------------------------------------------------------------

DROP TABLE IF EXISTS merge_decision   CASCADE;
DROP TABLE IF EXISTS notice_opportunity CASCADE;
DROP TABLE IF EXISTS opportunity_alias CASCADE;
DROP TABLE IF EXISTS opportunity      CASCADE;
DROP TABLE IF EXISTS lsh_bucket       CASCADE;
DROP TABLE IF EXISTS lsh_bucket_nc    CASCADE;
DROP TABLE IF EXISTS lsh_bucket_hash  CASCADE;
DROP TABLE IF EXISTS notice_bands     CASCADE;
DROP TABLE IF EXISTS notice_sketch    CASCADE;
DROP TABLE IF EXISTS notice           CASCADE;
DROP TABLE IF EXISTS index_meta       CASCADE;

-- 1 -------------------------------------------------------------------------
CREATE TABLE notice (
    notice_id       TEXT        PRIMARY KEY,
    portal_id       TEXT        NOT NULL,
    published_at    DATE        NOT NULL,
    title           TEXT        NOT NULL,
    body            TEXT        NOT NULL,
    estimated_value BIGINT,
    closing_date    DATE
);
CREATE INDEX notice_portal_idx ON notice (portal_id);

-- 2 -------------------------------------------------------------------------
-- The signature is stored as a fixed-width BYTEA: K little-endian uint32, the
-- low 32 bits of each MinHash. Fixed width is the point -- it is what lets the
-- row be banded and what keeps per-notice storage independent of body length.
CREATE TABLE notice_sketch (
    notice_id    TEXT        PRIMARY KEY REFERENCES notice(notice_id) ON DELETE CASCADE,
    k_rows       SMALLINT    NOT NULL,
    shingles     INTEGER     NOT NULL,
    sig          BYTEA       NOT NULL,
    sketched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3 -------------------------------------------------------------------------
-- The retrieval structure. One row per (notice, band). The lookup is always
--     WHERE band_no = ? AND bucket_key = ?
-- and always wants only notice_id back, which is why the primary key carries
-- notice_id as its third column: the index is then COVERING and the lookup
-- never touches the heap.
CREATE TABLE lsh_bucket (
    band_no     SMALLINT NOT NULL,
    bucket_key  BIGINT   NOT NULL,
    notice_id   TEXT     NOT NULL REFERENCES notice(notice_id) ON DELETE CASCADE,
    PRIMARY KEY (band_no, bucket_key, notice_id)
);

-- rejected alternative 1: the SAME B-tree, but not covering. Identical key
-- columns, notice_id left in the heap only. Isolates exactly what the third
-- column in the primary key buys.
CREATE TABLE lsh_bucket_nc (
    band_no     SMALLINT NOT NULL,
    bucket_key  BIGINT   NOT NULL,
    notice_id   TEXT     NOT NULL
);

-- rejected alternative 2: hash index
CREATE TABLE lsh_bucket_hash (
    band_no     SMALLINT NOT NULL,
    bucket_key  BIGINT   NOT NULL,
    notice_id   TEXT     NOT NULL
);

-- a second rejected alternative: one row per notice, all band keys in an array
CREATE TABLE notice_bands (
    notice_id   TEXT   PRIMARY KEY,
    keys        BIGINT[] NOT NULL
);

-- 4 -------------------------------------------------------------------------
-- A card id a bidder bookmarks must keep resolving. Opportunity ids are minted
-- once and never reissued; when two clusters turn out to be one, the younger
-- id becomes an alias of the older rather than disappearing.
CREATE TABLE opportunity (
    opportunity_id      TEXT        PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    canonical_notice_id TEXT,
    member_count        INTEGER     NOT NULL DEFAULT 0
);

CREATE TABLE notice_opportunity (
    notice_id      TEXT PRIMARY KEY REFERENCES notice(notice_id) ON DELETE CASCADE,
    opportunity_id TEXT NOT NULL REFERENCES opportunity(opportunity_id),
    assigned_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    join_score     REAL
);
CREATE INDEX notice_opportunity_opp_idx ON notice_opportunity (opportunity_id);

CREATE TABLE opportunity_alias (
    alias_id       TEXT PRIMARY KEY,
    opportunity_id TEXT NOT NULL REFERENCES opportunity(opportunity_id),
    merged_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE merge_decision (
    notice_id_a TEXT NOT NULL,
    notice_id_b TEXT NOT NULL,
    j_sketch    REAL,
    j_exact     REAL,
    decision    TEXT NOT NULL,          -- merge | review | reject
    decided_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (notice_id_a, notice_id_b)
);
CREATE INDEX merge_decision_decision_idx ON merge_decision (decision);

-- what the index was built with, so a restart can tell whether the buckets on
-- disk are still meaningful
CREATE TABLE index_meta (
    id              INT PRIMARY KEY DEFAULT 1,
    k_rows          INT  NOT NULL,
    bands           INT  NOT NULL,
    rows_per_band   INT  NOT NULL,
    variant         TEXT NOT NULL,
    scheme          TEXT NOT NULL,
    tau             REAL NOT NULL,
    tau_sketch      REAL NOT NULL,
    bucket_cap      INT,
    built_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
