\echo '--- tables in the deduplication schema ---'
\dt
\echo ''
\echo '--- row counts and on-disk size ---'
SELECT relname AS table,
       to_char(n_live_tup, '999,999,999') AS rows,
       pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_stat_user_tables ORDER BY pg_total_relation_size(relid) DESC LIMIT 10;
\echo ''
\echo '--- what the index was built with (survives a restart) ---'
SELECT k_rows, bands, rows_per_band, variant, scheme, tau, tau_sketch FROM index_meta;
\echo ''
\echo '--- the lookup the nightly job runs, on its access path ---'
EXPLAIN (ANALYZE, BUFFERS)
SELECT notice_id FROM lsh_bucket WHERE band_no = 12 AND bucket_key =
  (SELECT bucket_key FROM lsh_bucket WHERE band_no = 12 LIMIT 1);
\echo ''
\echo '--- a card, and the notices behind it ---'
SELECT o.opportunity_id, o.member_count, n.portal_id, left(n.title, 52) AS title
FROM opportunity o
JOIN notice_opportunity no ON no.opportunity_id = o.opportunity_id
JOIN notice n ON n.notice_id = no.notice_id
WHERE o.opportunity_id = (SELECT opportunity_id FROM notice_opportunity
                          GROUP BY opportunity_id ORDER BY count(*) DESC LIMIT 1)
ORDER BY n.notice_id;
\echo ''
\echo '--- merge decisions recorded tonight ---'
SELECT decision, count(*), round(avg(j_exact)::numeric, 4) AS mean_exact_jaccard
FROM merge_decision GROUP BY decision ORDER BY 2 DESC;
