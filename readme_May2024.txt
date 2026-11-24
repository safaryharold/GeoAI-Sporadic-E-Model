Radio occulation data used for sporadic E layer analysis have been downloaded from UCAR (https://data.cosmic.ucar.edu/gnss-ro/) in May 2024.
Sporadic E processing in May 2024. 

for every occultation measurement that reached at least the altitude of 110km above Earth's surface, the file contains information about
 doy,year,month,hour,minutes,lat,lon,alt,S4,isSporE
with: 

 doy:		 day of year 						(1-366)
 year: 											(2001-2023)
 month:		month of the year					(1-12)
 hour: 		time of RO event	in UTC 			(0-23)
 minutes: 	minutes of the respective hout in UTC (0-59)
 lat: 		geograph. latitude in °				(-90 - 90, 0° located at equator, negative values for southern hemisphere)
 lon: 		geograph. longitude in ° 			(-180 - 180, 0° located at Greenwich, negative values for western hemisphere)
 alt: 		height in km above Earth's surface of the point of highest SNR fluctuation (85 - 150)
 S4: 		S4 index, measure for disturbance degree in SNR profile, values > 0.2 indicate a disturbance (0-2)
 isSporE: 	value = 1 --> sporadic E layer was detected, value = 0 --> no sporadic E layer (0,1)


The file contains sporadic E layer information based on GNSS radio occultation data performed by several satellite missions.
Used data product and version:

mission		period						data product (UCAR)		version			#profiles (in 1000)			
CHAMP 		(2001 - 2008)				atmPhs					repro02016		393
GRACE 		(2007 - 2017) 				atmPhs					postProc		513
COSMIC-1 	(2007 - 2019)				conPhs					repro2021		6,562
COSMIC-2	(2020 - 2023) ongoing		conPhs					nrt				6,212
TerraSAR-X	(2008 - 2023) ongoing		atmPhs					postProc		960
PAZ			(2018 - 2023) ongoing		conPhs					postProc		202
TanDEM-X	(2016 - 2023) ongoing		conPhs					postProc		372
KompSat-5	(2019 - 2023) ongoing		conPhs					postProc		403
Spire		(2021 - 2023)*ongoing		conPhs					nrt				3,504
METOP-B		(2022 - 2023) ongoing		atmPhs					postProc		371
METOP-C		(2019 - 2023) ongoing		atmPhs					postProc		358
PlanetIQ	(2023 - 2024)				conPhs					nrt		 		396
																				total: 20,216


 
* data gap between summer 2023 and mid January 2024  

____________________________________________________________________________________________________
version news:
- PlanetIQ data are included from a short period end of 2023 - early 2024.  

