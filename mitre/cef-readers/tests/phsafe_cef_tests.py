from phsafe_safetab_reader.cef_reader import CEFReader
from pyspark.sql.functions import col, substring, concat

def test_pr_only():
    reader = CEFReader("src/phsafe_safetab_reader/cef_config.ini", ['72'])
    person_df = reader.get_person_df_input()
    unit_df = reader.get_unit_df_input()
    geo_df = reader.get_geo_df_input()

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == set(['72'])

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == set(['72'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['72'])

    print('puerto rico input test passed')

def test_us_only():
    reader = CEFReader("src/phsafe_safetab_reader/cef_config.ini", ['01', '02'])
    person_df = reader.get_person_df_input()
    unit_df = reader.get_unit_df_input()
    geo_df = reader.get_geo_df_input()

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == set(['01', '02'])

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == set(['01', '02'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['01', '02'])

    print('united states input test passed')

def test_pr_and_us():
    reader = CEFReader("src/phsafe_safetab_reader/cef_config.ini", ['72', '01', '02'])
    person_df = reader.get_person_df_input()
    unit_df = reader.get_unit_df_input()
    geo_df = reader.get_geo_df_input()

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == set(['72', '01', '02'])

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == set(['72', '01', '02'])

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == set(['72', '01', '02'])

    print('puerto rico and united states input test passed')

def test_allstates_and_pr():
    reader = CEFReader("src/phsafe_safetab_reader/cef_config.ini", [])
    person_df = reader.get_person_df_input()
    unit_df = reader.get_unit_df_input()
    geo_df = reader.get_geo_df_input()

    all_state_fips = set(['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56', '72'])

    per_states = [row.bcustatefp for row in person_df.select('bcustatefp').distinct().collect()]
    assert set(per_states) == all_state_fips

    unit_states = [row.bcustatefp for row in unit_df.select('bcustatefp').distinct().collect()]
    assert set(unit_states) == all_state_fips

    geo_states = [row.TABBLKST for row in geo_df.select('TABBLKST').distinct().collect()]
    assert set(geo_states) == all_state_fips

    print('puerto rico and all united states input test passed')

def test_input_types():
    reader = CEFReader("src/phsafe_safetab_reader/cef_config.ini", ['72', '01', '02'])
    person_df = reader.get_person_df_input()
    unit_df = reader.get_unit_df_input()
    geo_df = reader.get_geo_df_input()

    per_types = [row[1] for row in person_df.dtypes]
    assert per_types == ['string'] + 2 * ['bigint'] + ['string'] + ['bigint'] + ['string'] + 6 * ['bigint'] + ['string'] + 2 * ['bigint'] + 13 * ['string']

    unit_types = [row[1] for row in unit_df.dtypes]
    assert unit_types == ['string'] + ['bigint'] + ['string'] + ['bigint'] + 11 * ['string'] + ['bigint'] + 6 * ['string'] + 4 * ['bigint'] + 2 * ['string']

    geo_fields = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK', 'TABBLKSUFX1', 'TABBLKSUFX2', 'TABBLKGRPCE', 'POPDEC', 'HOUSING','CURSTATE','CURCOUNTY','CURTRACTCE', 'CURBLKGRPCE','REGIONCE','DIVISIONCE','STATENS','COUNTYNS','COUNTYFS','COUSUBFP','COUSUBNS','SUBMCDFP','SUBMCDNS','ESTATEFP','ESTATENS','CONCITFP','CONCITNS','PLACEFP','PLACENS','PLACEFS','AIANNHFP','AIANNHCE','AIANNHNS','AIHHTLI','TRIBALSUBFP','TRIBALSUBCE','TRIBALSUBNS','TTRACTCE','TBLKGRPCE','ANRCFP','ANRCNS','UACE','UATYP','UR','CD116FP','CDCURFP','VTDST','SLDUST','SLDLST','ZCTA5CE','SDELMLEA','SDSECLEA','SDUNILEA','UGACE','PUMA','LWBLKTYP','INTPTLAT','INTPTLON','AREALAND','AREAWATER','AREAWATERINLD','AREAWATERCSTL','AREAWATERGRLK','AREAWATERTSEA','CSAFP','CBSAFP','METDIVFP','PCICBSA','CNECTAFP','NECTAFP','NECTADIVFP','PCINECTA','ACT','MEMI','NMEMI','OIDTABBLK']
    geo_types = [row[1] for row in geo_df.dtypes if row[0] in geo_fields]
    assert geo_types == 74 * ['string'] + ['bigint']

    print('input types test passed')

def test_output_types_and_order():
    reader = CEFReader("src/phsafe_safetab_reader/cef_config.ini", ['72', '01', '02'])

    person_df = reader.get_person_df()
    unit_df = reader.get_unit_df()
    geo_df = reader.get_geo_df()

    per_types = [row for row in person_df.dtypes]
    assert per_types == [('RTYPE', 'string'), ('MAFID', 'bigint'), ('QAGE', 'bigint'), ('CENHISP', 'bigint'), ('CENRACE', 'string'), ('RELSHIP', 'string')]

    unit_types = [row for row in unit_df.dtypes]
    assert unit_types == [('RTYPE', 'string'), ('MAFID', 'bigint'), ('FINAL_POP', 'bigint'), ('NPF', 'bigint'), ('HHSPAN', 'bigint'), ('HHRACE', 'string'), ('TEN', 'string'), ('HHT', 'string'), ('HHT2', 'string'), ('CPLT', 'string')]

    geo_fields = ['RTYPE', 'MAFID', 'TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK','TABBLKGRPCE','REGIONCE','DIVISIONCE','PLACEFP','AIANNHCE']
    geo_types = [row[1] for row in geo_df.dtypes if row[0] in geo_fields]
    assert geo_types == ['string'] + ['bigint'] + 9 * ['string']

    print('output types and order test passed')

def run_all_tests():
    test_pr_only()
    test_us_only()
    test_pr_and_us()
    test_allstates_and_pr()
    test_input_types()
    test_output_types_and_order()

if __name__ == "__main__":
    run_all_tests()
