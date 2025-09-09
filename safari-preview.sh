open -a safari --background http://127.0.0.1:23635
hx $1
osascript -e "tell application \"Safari\" to quit"
