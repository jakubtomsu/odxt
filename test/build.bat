cl /c /DNDEBUG stb_dxt.c
@REM cl /FA /c /DNDEBUG stb_dxt.c
lib /OUT:stb_dxt.lib stb_dxt.obj
del stb_dxt.obj