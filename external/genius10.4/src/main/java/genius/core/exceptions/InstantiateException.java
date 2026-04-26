package genius.core.exceptions;

/**
 * Indicates that no instance of some object could be created.
 * 
 * @author W.Pasman 27jul15
 *
 */
@SuppressWarnings("serial")
public class InstantiateException extends Exception {

	public InstantiateException(String string) {
		super(string);
	}

	public InstantiateException(String string, Exception e) {
		super(string, e);
	}

}
