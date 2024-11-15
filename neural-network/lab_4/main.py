from model import AlexNet


alex_net = AlexNet()
alex_net.load()
answ = alex_net.predict('/Users/sayner/github_repos/neural-network/lab_5/files/test_img_6.jpg')
print(answ)