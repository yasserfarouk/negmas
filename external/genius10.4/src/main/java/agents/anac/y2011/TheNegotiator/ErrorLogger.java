package agents.anac.y2011.TheNegotiator;

import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

/**
 * Simple class used to log errors in a logfile.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class ErrorLogger {
	
	// used to ensure that the ErrorLogger is only instantiated once.
	// This function was designed so that a startfunction would not be
	// required.
	static boolean initialized = false;
	// reference to the logger which stores the errors into a log
	static Logger logger;
	
	/**
	 * Used to log an error message in xml-format in a html file.
	 * Messages are appended.
	 * 
	 * @param error message which should be stored
	 */
	public static void log(String error) {
		if (!initialized) {
			logger = Logger.getLogger("MyLog");
			FileHandler fh;
		
			try {
				// This block configure the logger with handler and formatter
				fh = new FileHandler("Errors.html", true);
				logger.addHandler(fh);
				logger.setLevel(Level.ALL);
				SimpleFormatter formatter = new SimpleFormatter();
				fh.setFormatter(formatter);
		
				// the following statement is used to log any messages   
				
			} catch (Exception e) {
				e.printStackTrace();
			}
				initialized = true;
		}
		logger.log(Level.INFO, error + "<br></br>");
	}
} 