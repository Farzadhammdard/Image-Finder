#define MyAppName "Image Finder"
#define MyAppVersion "4.0.0"
#define MyAppDisplayVersion "v-4.0.0"
#define MyAppPublisher "Engineer Ahmad Farzad Hamdard"
#define MyAppExeName "ImageFinder.exe"
#ifndef MySourceDir
  #define MySourceDir "..\dist_v4\ImageFinder"
#endif

[Setup]
AppId={{24EEA7DE-13D8-4AB0-B0F7-8CF2FC7008A6}
AppName={#MyAppName}
AppVerName={#MyAppName} {#MyAppDisplayVersion}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppComments=Designed by Engineer Ahmad Farzad Hamdard
DefaultDirName={autopf}\ImageFinder
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=ImageFinderSetup
SetupIconFile=..\assets\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
