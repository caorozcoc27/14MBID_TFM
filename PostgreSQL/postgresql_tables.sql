
--------------------------
-- TFM Postgresql Tables
--------------------------

-- enrichmentdata

CREATE TABLE enrichmentdata (
	neId varchar(255) PRIMARY KEY,
	neName varchar(255),
	neManufacturer varchar(255),
	neModel varchar(255),
	operativeState varchar(255),
	comunityName varchar(255),
	provinceName varchar(255),
	cityName varchar(255),
	postalCode varchar(255),
	locationId varchar(255),
	latitude real,
	longitude real	 
);

INSERT INTO enrichmentdata VALUES
	('BCN014_01', 'CELL_BCN014_01', 'Huawei', 'gNBU2020', 'Operative', 'Cataluña', 'Barcelona', 'Barcelona', '08032', 'BCN014', 41.42702537, 2.15492885),
	('TAR015_01', 'CELL_TAR015_01', 'Huawei', 'gNBU2000', 'Operative', 'Cataluña', 'Tarragona', 'Tarragona', '43005', 'TAR015', 41.11834909, 1.24486151),
	('MAD033_01', 'CELL_MAD033_01', 'Huawei', 'gNBU2020', 'Operative', 'Madrid', 'Madrid', 'Madrid', '28002', 'MAD033', 40.40146445, -3.69136465),
	('TOL034_01', 'CELL_TOL034_01', 'Huawei', 'gNBU2000', 'Operative', 'Castilla La Mancha', 'Toledo', 'Toledo', '45003', 'TOL034', 39.85734838, -4.02576328),
	('SAL030_01', 'CELL_SAL030_01', 'Huawei', 'gNBU2000', 'Operative', 'Castilla y Leon', 'Salamanca', 'Salamanca', '37007', 'SAL030', 40.97113447, -5.66105771),
	('VAL002_01', 'CELL_VAL002_01', 'Huawei', 'gNBU2020', 'Operative', 'Comunidad Valenciana', 'Valencia', 'Valencia', '46017', 'VAL002', 39.47212863, -0.37869797),
	('BIL001_01', 'CELL_BIL001_01', 'Huawei', 'gNBU2010', 'Operative', 'Pais Vasco', 'Vizcaya', 'Bilbao', '48008', 'BIL001', 43.26746297, -2.93354724),
	('SEV001_01', 'CELL_SEV001_01', 'Huawei', 'gNBU2000', 'Operative', 'Andalucia', 'Sevilla', 'Sevilla', '41001', 'SEV001', 37.38574508, -5.98572065),
	('SNT002_01', 'CELL_SNT002_01', 'Huawei', 'gNBU2000', 'Operative', 'Galicia', 'La Coruña', 'Santiago', '15702', 'SNT002', 42.87462078, -8.55478157);


-- metricslist

CREATE TABLE metricslist (
	kpiId numeric PRIMARY KEY,
	kpiName	varchar(255),
	counterNum	varchar(255),
	counterDen	varchar(255),
	category	varchar(255),
	vendor	varchar(255),
	unit	varchar(255),
	enabled	boolean
);

INSERT INTO metricslist VALUES
	(1, 'DLPDCP_VOLUME', 'INTEGRITY_NGBR_QCI_9', '', 'Integrity', 'Huawei', 'simple', true),
	(2, 'VOICE_TRAFFIC', 'VOICE_TRAFFIC', '', 'Integrity', 'Huawei', 'simple', true),
	(3, 'SERVICE_LATENCY_OVER_AIR_INTERFACE', 'SERVICE_LATENCY_OVER_AIR_INTERFACE_NUM', 'SERVICE_LATENCY_OVER_AIR_INTERFACE_DEN', 'Integrity', 'Huawei', 'rate', true),
	(4, 'INTRA_FREQUENCY_HANDOVER_SUCCESS_RATE', 'INTRA_FREQUENCY_HO_OUT_SUCC', 'INTRA_FREQUENCY_HO_OUT_ATT', 'Mobility', 'Huawei', 'rate', true),
	(5, 'ERAB_SUCCESS_RATE', 'ERAB_SR_NUM', 'ERAB_SR_DEN', 'Retainability', 'Huawei', 'rate', true),
	(6, 'DROP_CALL_RATE', 'DCR_NUM', 'DCR_DEN', 'Retainability', 'Huawei', 'rate', true),
	(7, 'VOLTE_DROP_CALL_RATE', 'DROP_CALL_VOLTE_NUM', 'DROP_CALL_VOLTE_DEN', 'Retainability', 'Huawei', 'rate', true),
	(8, 'RRC_SETUP_SUCCESS_RATE', 'RRC_SETUP_SUCC', 'RRC_SETUP_ATT', 'Accessibility', 'Huawei', 'rate', true),
	(9, 'VOLTE_CALL_SETUP_SUCCESS_RATE', 'VOLTE_CALL_SETUP_SUCC', 'VOLTE_CALL_SETUP_ATT', 'Accessibility', 'Huawei', 'rate', true),
	(10, 'S1SIG_CONNECTION_SUCCESS_RATE', 'S1SIG_Connection_Setup_SR_NUM', 'S1SIG_Connection_Setup_SR_DEN', 'Accessibility', 'Huawei', 'rate', true);


COMMIT;