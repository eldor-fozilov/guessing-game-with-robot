



@ WIN
usbipd.exe list
wsl -l


usbipd.exe attach --busid <번호>
usbipd.exe attach --busid <번호> --wls <WSL 중 하나>

ex) usbipd.exe attach --busid 7-4 
ex) usbipd.exe attach --busid 7-4 --wsl Ubuntu

