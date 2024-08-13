 Runtime Performance (Logging)
	Takes a string and logs it to the runtime performance logfile along with running times
	Usage:
		from utils import PerformanceLogger
		plog = PerformanceLogger()
		  Options:
			filepath		str	 the folder you want to store the log in, defaults to ./
			filename		str	 the log's filename, defaults to performance_runtime.log
			to_screen	   bool	output to screen as well, defaults True
			enable_logging  bool	to allow disabling the log for squeezing out performance

		plog = PerformanceLogger()
		plog.perf_go("My Speedy App")					   # Start your engines, inits the timer
		plog.perf_lap(f"Finished Loop {loop_counter}")	  # mark points in code you wish to count as a lap. Considers the code before it, not after, so put at the end of code you want to profile.
		plog.perf_end("All Done")						   # end the performance counter, reports all of the laptimes

		### You can also log some intersting events along with runtime module information
		plog.it("I've just started someting interesting")
		plog.it("I've just finished something interesting")

		### You can surpress or focus on particular modules by just adding their names to the filter
		plog.surpress_modules(["spammy_function_1", "spammy_module_2"])			 # stop these log messages being sent to screen or logfile
		plog.focus_modules(["interesting_function_1", "interesting_function_2"])	# stop all output *except these*

