ALTER TABLE signs_of_life_crawler 
ADD COLUMN dns_has_dmarc VARCHAR,
ADD COLUMN dns_value_dmarc VARCHAR,
ADD COLUMN dns_dmarc_comment VARCHAR;