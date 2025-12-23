function ClickConnect(){
    console.log("Working... 1分ごとに接続確認中"); 
    document.querySelector("colab-connect-button").click() 
}
setInterval(ClickConnect, 60000)