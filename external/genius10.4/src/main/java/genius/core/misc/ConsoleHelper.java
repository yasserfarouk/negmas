package genius.core.misc;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

/**
 * Created by david on 19/07/15.
 */
public class ConsoleHelper {

	private static PrintStream orgErr, orgOut = null;

	/**
	 * Silences or restores the console output. This can be useful to suppress
	 * output of foreign code, like submitted agents
	 *
	 * @param enable
	 *            Enables console output if set to true or disables it when set
	 *            to false
	 */
	public static void useConsoleOut(boolean enable) {

		if (orgErr == null)
			orgErr = System.err;
		if (orgOut == null)
			orgOut = System.out;

		if (enable) {
			System.setErr(orgErr);
			System.setOut(orgOut);
		} else {
			System.setOut(new PrintStream(new OutputStream() {
				@Override
				public void write(int b) throws IOException {
				}
			}));
			System.setErr(new PrintStream(new OutputStream() {
				@Override
				public void write(int b) throws IOException {
				}
			}));
		}
	}
}
