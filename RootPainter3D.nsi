;Copyright (C) 2022 André Pedersen

;This program is free software: you can redistribute it and/or modify
;it under the terms of the GNU General Public License as published by
;the Free Software Foundation, either version 3 of the License, or
;(at your option) any later version.

;This program is distributed in the hope that it will be useful,
;but WITHOUT ANY WARRANTY; without even the implied warranty of
;MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;GNU General Public License for more details.

;You should have received a copy of the GNU General Public License
;along with this program.  If not, see <https://www.gnu.org/licenses/>.

!define APP_NAME "RootPainter3D"
!define COMP_NAME "University of Copenhagen"
!define VERSION "0.0.1"
!define DESCRIPTION "Application"
!define INSTALLER_NAME "RootPainter3D-0.0.1-win.exe"
!define MAIN_APP_EXE "RootPainter3D.exe"
!define INSTALL_TYPE "SetShellVarContext current"
!define REG_ROOT "HKLM"

!define REG_APP_PATH "Software\Microsoft\Windows\CurrentVersion\App Paths\${MAIN_APP_EXE}"
!define UNINSTALL_PATH "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
!define REG_START_MENU "Start Menu Folder"

var SM_Folder

######################################################################

SetCompressor ZLIB
Name "${APP_NAME}"
Caption "${APP_NAME}"
OutFile "${INSTALLER_NAME}"
BrandingText "${APP_NAME}"
XPStyle on
InstallDirRegKey "${REG_ROOT}" "${REG_APP_PATH}" ""
InstallDir "$PROGRAMFILES\RootPainter3D"

!include 'MUI.nsh'
!define MUI_ICON ".\painter\icons\icon.ico"

!define MUI_ABORTWARNING
!define MUI_UNABORTWARNING

!insertmacro MUI_PAGE_WELCOME

!insertmacro MUI_PAGE_DIRECTORY

!ifdef REG_START_MENU
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "RootPainter3D"
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "${REG_ROOT}"
!define MUI_STARTMENUPAGE_REGISTRY_KEY "${UNINSTALL_PATH}"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "${REG_START_MENU}"
!insertmacro MUI_PAGE_STARTMENU Application $SM_Folder
!endif

!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

# name the installer
OutFile "${INSTALLER_NAME}"

Section
SectionEnd


####################### UNINSTALL BEFORE UPGRADE #####################
Section "" SecUninstallPrevious
    Call UninstallPrevious
SectionEnd

Function UninstallPrevious
    ; Check for uninstaller.
    DetailPrint "Checking for previous RootPainter3D versions"
    ReadRegStr $R0 HKCU "$INSTDIR" "UninstallString"

    ${If} $R0 == ""
         ReadRegStr $R0 HKLM "$INSTDIR" "UninstallString"
        ${If} $R0 == ""
            DetailPrint "No previous installation found"
            Goto Done
        ${EndIf}
    ${EndIf}

    DetailPrint "Removing previous installation."
    ; Run the uninstaller silently.
    ExecWait '"$R0" /S _?=$INSTDIR' $0
    DetailPrint "Uninstaller returned $0"
    Done:
FunctionEnd
######################################################################

# default section start; every NSIS script has at least one section.

######################################################################
Section -MainProgram
${INSTALL_TYPE}
SetOverwrite ifnewer
SetOutPath "$INSTDIR"
SectionEnd
######################################################################

Section -Icons_Reg
SetOutPath "$INSTDIR"
WriteUninstaller "$INSTDIR\uninstall.exe"

!ifdef REG_START_MENU
!insertmacro MUI_STARTMENU_WRITE_BEGIN Application
CreateDirectory "$SMPROGRAMS\$SM_Folder"
CreateShortCut "$SMPROGRAMS\$SM_Folder\${APP_NAME}.lnk" "$INSTDIR\${MAIN_APP_EXE}" "" "$INSTDIR\painter\icons\icon.ico" 0
CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${MAIN_APP_EXE}" "" "$INSTDIR\painter\icons\icon.ico" 0
CreateShortCut "$SMPROGRAMS\$SM_Folder\Uninstall ${APP_NAME}.lnk" "$INSTDIR\uninstall.exe"

!insertmacro MUI_STARTMENU_WRITE_END
!endif

# define the output path for this file
SetOutPath $INSTDIR

WriteRegStr ${REG_ROOT} "${REG_APP_PATH}" "" "$INSTDIR\${MAIN_APP_EXE}"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "DisplayName" "${APP_NAME}"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "UninstallString" "$INSTDIR\uninstall.exe"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "DisplayIcon" "$INSTDIR\painter\icons\icon.ico"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "DisplayVersion" "${VERSION}"
WriteRegStr ${REG_ROOT} "${UNINSTALL_PATH}"  "Publisher" "${COMP_NAME}"

# Create directory
CreateDirectory $INSTDIR

# PACKAGE ENTIRE CONTENT OF BUNDLE THE NEW BINARY!
File /nonfatal /a /r ".\dist\RootPainter3D\*"
ExecWait "$INSTDIR\RootPainter3D-installed.exe"

# default section end
SectionEnd

######################################################################

Section Uninstall
${INSTALL_TYPE}
RmDir /r "$INSTDIR"

!ifdef REG_START_MENU
!insertmacro MUI_STARTMENU_GETFOLDER "Application" $SM_Folder
Delete "$SMPROGRAMS\$SM_Folder\${APP_NAME}.lnk"
Delete "$SMPROGRAMS\$SM_Folder\Uninstall ${APP_NAME}.lnk"
Delete "$DESKTOP\${APP_NAME}.lnk"

RmDir "$SMPROGRAMS\$SM_Folder"
!endif

DeleteRegKey ${REG_ROOT} "${REG_APP_PATH}"
DeleteRegKey ${REG_ROOT} "${UNINSTALL_PATH}"
SectionEnd

######################################################################
