package genius.core.misc;

public class ExceptionTool {
	private Throwable exception;

	public ExceptionTool(Throwable e) {
		exception = e;
	}

	/**
	 * @return the full message for the exception, including all sub-causes.
	 */
	public String getFullMessage() {
		Throwable e = exception;
		String msg = "";

		while (e != null) {
			msg = msg + (msg.isEmpty() ? "" : ": ") + e.getMessage();
			e = e.getCause();
		}
		return msg;
	}
}
