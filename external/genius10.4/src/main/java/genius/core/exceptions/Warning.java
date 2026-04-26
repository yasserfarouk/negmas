package genius.core.exceptions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;

/**
 * Warning objects handle warning messages. These objects also count how many
 * times a particular type of message has been issued already. You can ask for a
 * stack dump as well.
 */

public class Warning {

	public final static int DEFAULT_SUPPRESSION_NUMBER = 5;

	protected class MyWarningException extends Exception {

		private static final long serialVersionUID = 2047098752954743217L;
	}

	/*
	 * Hashtable key = warning message, corresponding value = #repetitions.
	 */
	static Hashtable<String, Integer> pPreviousMessages = new Hashtable<String, Integer>();

	/**
	 * Default warning: Print warning message at most 5 times. Stack trace is
	 * not printed.
	 */
	public Warning(String warning) {
		makeWarning(warning, new MyWarningException(), false, DEFAULT_SUPPRESSION_NUMBER);
	}

	/**
	 * The location of the error will be reported as the code location where
	 * this warning is placed. Note that this is not useful if you are
	 * converting an exception into a warning. In that case, you better use
	 * Warning(warning, Exception).
	 * 
	 * @param warning
	 *            is the message to be shown
	 * @param showstack
	 *            is true if you want to show a stack dump as well. Then, stack
	 *            dump will be made for location where WARNING occurs
	 * @param suppressat
	 *            is the maximum number of this warning you want to appear
	 */
	public Warning(String warning, boolean showstack, int suppressat) {
		makeWarning(warning, new MyWarningException(), showstack, suppressat);
	}

	public Warning(String pWarning, Exception err) {
		makeWarning(pWarning, err, false, DEFAULT_SUPPRESSION_NUMBER);
	}

	/**
	 * Note that this is not useful if you are converting an exception into a
	 * warning. In that case, you better use Warning(warning, Exception)
	 * 
	 * @param pWarning
	 *            is the message to be shown
	 * @param err
	 *            is the exception that caused the rise of this warning. this
	 *            will be used to inform the user about where the problem
	 *            occured.
	 * @param pShowStack
	 *            is true if you want to show a stack dump as well. If set,
	 *            stack dump will be made for location where WARNING occurs.
	 * @param pSuppressAt
	 *            is the maximum number of this warning you want to appear
	 */
	public Warning(String pWarning, Exception err, boolean pShowStack, int pSuppressAt) {
		makeWarning(pWarning, err, pShowStack, pSuppressAt);
	}

	/**
	 * Add warning to static hashtable used to keep track of all warnings issued
	 * so far. Only show warning if message has not appeared more than
	 * 'fSuppressAt' times.
	 * 
	 * @param e
	 *            is exception that caused the problem. Use null to avoid stack
	 *            dump.
	 */
	public void makeWarning(String pWarning, Exception e, boolean pDumpStack, int pSuppressAt) {

		Object lWarnings = pPreviousMessages.get(pWarning);

		if (lWarnings == null) {
			pPreviousMessages.put(pWarning, 0);
			lWarnings = 0;
		}

		int lNrOfWarnings = (Integer) (pPreviousMessages.get(pWarning)) + 1;
		// Update nr of warning occurrences in hashtable
		pPreviousMessages.put(pWarning, lNrOfWarnings);

		if ((Integer) lWarnings > pSuppressAt)
			return;

		System.out.print("WARNING: " + pWarning + ", " + e);

		ArrayList<StackTraceElement> elts = new ArrayList<StackTraceElement>(Arrays.asList(e.getStackTrace()));
		ArrayList<StackTraceElement> tmp = new ArrayList<StackTraceElement>(elts);
		if (e instanceof MyWarningException) {

			tmp.remove(0); // remove the warning itself from the trace.
		}
		while ((!tmp.isEmpty()) && (tmp.get(0).toString().indexOf(':') == -1))
			tmp.remove(0);
		if (tmp.isEmpty())
			tmp = elts;

		if (pDumpStack) {
			System.out.println();
			for (StackTraceElement elt : tmp)
				System.out.println(elt);
		} else {
			if (!(tmp.isEmpty()))
				System.out.print(" at " + tmp.get(0) + "\n");
			else
				System.out.print(" at empty stack point?\n");
		}

		if ((Integer) lWarnings == pSuppressAt) {
			System.out.print("New occurrences of this warning will not be shown anymore.\n");
			return;
		}
	}

}
