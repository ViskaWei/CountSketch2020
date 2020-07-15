import getpass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import umap
import os
from collections import Counter
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
from SciServer import Authentication, CasJobs
username = 'anyon'

password = getpass.getpass()
sciserver_token = Authentication.login(username, password)
sciserver_token
# https://github.com/sciserver/SciScript-Python/blob/master/Examples/Examples_SciScript-Python.ipynb

result = CasJobs.uploadPandasDataFrameToTable(dataFrame=df, tableName=CasJobs_TestTableName2, context="MyDB")
sql_n,ii=20,0
sql0="""
       select
               {}
              objid, class, subclass, u-g as ug, g-r as gr, r-i as ri, i-z as iz, 
              u-r as ur, g-i as gi, r-z as rz,
              u-i as ui, g-z as gz, u-z as uz,  (z-15)/7 as z 
from (
       SELECT p.objid, s.class AS class, s.SUBCLASS AS subclass,          
            p.psfMag_u AS u, 
            p.psfMag_g AS g, 
            p.psfMag_r AS r, 
            p.psfMag_i AS i, 
            p.psfMag_z AS z
        FROM PhotoObj as P join SpecObjAll as S on P.objID = s.BestObjID
        WHERE p.objid in (select objid from MYDB.photosampleDR13)
) x
{}
order by 1
OFFSET {} ROWS FETCH NEXT {} ROWS ONLY;""".format('', '',sql_n*ii,sql_n)


sql_QSG="""
        SELECT {} 
        TOP 1000000
            class AS class,SUBCLASS AS subclass,     
            p.extinction_u AS ext_u, p.psfMag_u AS u, p.dered_u AS der_u, 
            p.extinction_g AS ext_g, p.psfMag_g AS g, p.dered_g AS der_g, 
            p.extinction_r AS ext_r, p.psfMag_r AS r, p.dered_r AS der_r, 
            p.extinction_i AS ext_i, p.psfMag_i AS i, p.dered_i AS der_i,
            p.extinction_z AS ext_z, p.psfMag_z AS z, p.dered_z AS der_z,
            p.probPSF_u AS prob, p.probPSF_r AS prob_r
        INTO MYDB.NIPS15
        FROM PhotoObj as P left outer join SpecObjAll as S on P.objID = s.BestObjID
        WHERE p.type=6 AND p.clean=1
             AND p.psfMag_u between 15 and 22
             AND p.psfMag_g between 15 and 22 
             AND p.psfMag_r between 15 and 20 
             AND p.psfMag_i between 15 and 22
             AND p.psfMag_z between 15 and 22
              {}
         ORDER BY objID, r,u,g,i,z, class,subclass """.format('', '')

sql_Qspec="""
        SELECT {}
            class AS class,SUBCLASS AS subclass,            
            p.extinction_u AS ext_u, p.psfMag_u AS u, p.dered_u AS der_u, 
            p.extinction_g AS ext_g, p.psfMag_g AS g, p.dered_g AS der_g, 
            p.extinction_r AS ext_r, p.psfMag_r AS r, p.dered_r AS der_r, 
            p.extinction_i AS ext_i, p.psfMag_i AS i, p.dered_i AS der_i,
            p.extinction_z AS ext_z, p.psfMag_z AS z, p.dered_z AS der_z,
            p.probPSF_u AS prob, p.probPSF_r AS prob_r
        FROM SpecObj s
            INNER JOIN sppParams spp ON spp.specobjID = s.specObjID
            INNER JOIN PhotoObj p ON p.objID = s.bestObjID
        WHERE type = 6 AND class='QSO' AND p.clean=1
             AND p.psfMag_u between 15 and 22
             AND p.psfMag_g between 15 and 22 
             AND p.psfMag_r between 15 and 20 
             AND p.psfMag_i between 15 and 22
             AND p.psfMag_z between 15 and 22
              {}
        ORDER BY class,subclass, u,g,r,i,z""".format('', '')


sql_spec='''
select  class, subclass, u-g as ug, g-r as gr, r-i as ri, i-z as iz, u-r as ur, g-i as gi, r-z as rz, u-i as ui, g-z as gz, u-z as uz,  (r-15)/5 as r,objid 
--into MYDB.spectrain
from (SELECT p.psfMag_u AS u, p.psfMag_g AS g, p.psfMag_r AS r, p.psfMag_i AS i, p.psfMag_z AS z,
s.class as class, s.subclass as subclass,s.BestObjID as objid
        FROM SpecObj as s left join PhotoObj as P  on s.BestObjID= P.objID
        WHERE s.BestObjID in (select objid from MYDB.specDR13) 
) x 
order by 1'''

sql_drop = """IF OBJECT_ID('spectrain') IS NOT NULL
        DROP TABLE spectrain"""
CasJobs.executeQuery(sql=sql_drop, context='MYDB', format="pandas")

sql_get="""
    select * from MYDB.NIPSspecphoto order by objid
    {} 
    OFFSET {} ROWS FETCH NEXT 1407000 ROWS ONLY;""".format('',1000000*ii)
# ii+=1

df = CasJobs.executeQuery(sql='SELECT * FROM N15', context='MYDB', format="pandas")
jobid = CasJobs.submitJob(sql=sql_spec, context='DR13')
CasJobs.waitForJob(jobid)

df = CasJobs.executeQuery(sql=sql_spec, context='DR13', format="pandas")
df = CasJobs.executeQuery(sql='SELECT * FROM MYDB.spectrain', context='MYDB', format="pandas")
df.shape

df.to_csv('../data/sdss_stars/DR13/544k_spec_objid.csv',index=False)

'''
select  class, subclass, u-g as ug, g-r as gr, r-i as ri, i-z as iz, u-r as ur, g-i as gi, r-z as rz, u-i as ui, g-z as gz, u-z as uz,  
  (u-15)/7 as u, (g-15)/7 as g, (i-15)/7 as i,  (z-15)/7 as z, (r-15)/5 as r,
  
  u-0.5*ext_u as u05,  g-0.5*ext_g as g05,  r-0.5*ext_r as r05,  i-0.5*ext_i as i05, z-0.5*ext_z as z05,
  
  u-0.25*ext_u as u25, g-0.25*ext_g as g25, r-0.25*ext_r as r25, i-0.25*ext_i as i25,z-0.25*ext_z as z25,
  
  u-0.75*ext_u as u75, g-0.75*ext_g as g75, r-0.75*ext_r as r75, i-0.75*ext_i as i75,z-0.75*ext_z as z75

into MYDB.specext
from (SELECT p.extinction_u AS ext_u, p.psfMag_u AS u,  
            p.extinction_g AS ext_g, p.psfMag_g AS g, 
            p.extinction_r AS ext_r, p.psfMag_r AS r, 
            p.extinction_i AS ext_i, p.psfMag_i AS i, 
            p.extinction_z AS ext_z, p.psfMag_z AS z,
            s.class as class, s.subclass as subclass
        FROM PhotoObj as p left join SpecObj as s on P.objID=s.BestObjID
        WHERE P.objID in (select objid from MYDB.specDR13) 
) x 
order by 1
'''