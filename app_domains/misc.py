from config import RUN_CONFIG

def VDM(output):
    if RUN_CONFIG["VERBOSE_DEBUGGING"] == True:
        print(f"******** {output} ********")
