from model import AlexNet


alex_net = AlexNet()
alex_net.load()
answ = alex_net.predict('/Users/sayner/github_repos/data-science/neural-network/inception_v3/files/test_img_6.jpg')
print(answ)
