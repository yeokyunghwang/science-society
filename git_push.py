git remote set-url origin https://github.com/yeokyunghwang/science-society.git
git remote -v


apt-get update
apt-get install -y git-lfs
git lfs install
git lfs version


cd /root/science-society
git add .
git commit -m "update"
git push
