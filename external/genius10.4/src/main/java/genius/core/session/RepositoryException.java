package genius.core.session;

/**
 * Error that will be thrown when we fail to fetch data from repository XML
 * files. RuntimeException because this is a configuration problem.
 */
public class RepositoryException extends RuntimeException {
	public RepositoryException(String message, Throwable cause) {
		super(message + ". This is caused by missing/damaged basic configuration xml files from Genius.", cause);
	}

}
