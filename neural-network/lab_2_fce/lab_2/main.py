from lab_2_fce.lab_2.tools import network_analyzer

CurrentNetworkAnalyzer = network_analyzer.NetworkAnalyzer()
CurrentNetworkAnalyzer.fit_networks(10, 0.05, 1, 100)

# CurrentNetworkAnalyzer = network_analyzer.NetworkAnalyzer()
# for i in range(10):
#     start = time.time()
#     CurrentNetworkAnalyzer.fit_networks(100, 0.05, 1, 200)
#     print(f"executing time : {time.time() - start:.0f} secs")
#     print("sleeping...\n")
#     time.sleep(600)
# CurrentNetworkAnalyzer.print_resulting_plots(100, 40)
