@echo off
chcp 65001 >nul
title 柑橘实例分割实验智能总结与出图工作台
echo ========================================================
echo   🍊 柑橘实验全自动总结、出图与报告生成工作台
echo ========================================================
echo.

set PYTHON_EXE=python
where python >nul 2>nul
if %errorlevel% neq 0 (
    if exist "E:\AppInstallion\0_4_annaconda\python.exe" (
        set PYTHON_EXE=E:\AppInstallion\0_4_annaconda\python.exe
    )
)

echo 正在扫描并分析最近的实验结果...
echo.

%PYTHON_EXE% "%~dp0summarize_experiments.py" --open

echo.
echo ========================================================
echo   🎉 处理完成！报告与图表已生成，并在浏览器中自动打开！
echo ========================================================
echo.
pause
