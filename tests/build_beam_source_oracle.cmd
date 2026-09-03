@echo off
setlocal
rem Run from the TPUBeamSearch root; the reference checkout is read-only.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
if not exist .local mkdir .local
cl /nologo /EHsc /std:c++17 /DBEAM_STATE_LOGICAL_BYTES=120 /DBEAM_STATE_PHYSICAL_BYTES=128 /DBEAM_STATE_ALIGNMENT=16 /DBEAM_MOVE_COUNT=24 /I D:\100XH100\src tests\beam_source_oracle.cpp D:\100XH100\src\hash.cpp D:\100XH100\src\state.cpp D:\100XH100\src\stream4.cpp /Fe:.local\beam_source_oracle.exe /Fo:.local\
exit /b %errorlevel%
