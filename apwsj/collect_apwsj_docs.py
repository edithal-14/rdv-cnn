qimport os
rootdir="TIPSTER"
collection_dir="/home/development/tirthankar/tirthankar/apwsj/apwsj_documents"
if not os.path.exists(collection_dir):
	os.makedirs(collection_dir)
for dirname,subdirlist,filelist in os.walk(rootdir):
	if dirname.endswith("AP"):
		print("Processing directory "+dirname)
		if len([f for f in os.listdir(dirname) if f.endswith(".Z")])>0:
			os.system("uncompress "+dirname+"/*.Z")
		os.system("cp "+dirname+"/AP* "+collection_dir+"/")
	elif dirname.endswith("WSJ"):
		for dir in subdirlist:
			print("Processing directory "+dirname+"/"+dir)
			if len([f for f in os.listdir(dirname+"/"+dir) if f.endswith(".Z")])>0:
				os.system("uncompress "+dirname+"/"+dir+"/*.Z")
			os.system("cp "+dirname+"/"+dir+"/WSJ* "+collection_dir+"/")
