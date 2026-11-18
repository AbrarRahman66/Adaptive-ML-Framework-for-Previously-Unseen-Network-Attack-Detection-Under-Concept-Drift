-- ADDITIVE migration for Rayan's `nids` database (MariaDB 10.4+).
-- Only adds NEW nullable columns, so his existing queries and data keep working.
-- Safe to run more than once. main.py also runs it on startup (AUTO_MIGRATE=1).

ALTER TABLE `predictions`
  ADD COLUMN IF NOT EXISTS `rf_pred`          TINYINT(1)  DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS `rf_prob_attack`   FLOAT       DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS `xgb_pred`         TINYINT(1)  DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS `xgb_prob_attack`  FLOAT       DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS `recon_error`      DOUBLE      DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS `anomaly_flag`     TINYINT(1)  DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS `final_verdict`    VARCHAR(30) DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS `created_at`       TIMESTAMP   NOT NULL DEFAULT current_timestamp();
