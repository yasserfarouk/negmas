package genius.core.listener;

import javax.swing.SwingUtilities;

/**
 * Listener for changes in a {@link Listenable}.
 *
 * @param <T>
 *            type of data being notified back to the caller.
 */
public interface Listener<T> {
	/**
	 * a notification call that occurs when something changed. Consult the
	 * parent object for details about this event.
	 * 
	 * <h1>NOTICE 1</h1> notifications run in the thread of the caller. The
	 * caller will be blocked until this callback has been completed. It is
	 * therfore good practice to handle callbacks quickly.
	 * 
	 * <h1>NOTICE 2</h1> Notifications often cross a thread sarety boundary. For
	 * example when in MVC a model (typically thread safe) calls back a panel
	 * (not thread safe). The called side (the panel) thus must handle the
	 * callback in a thread safe manner, e.g. using
	 * {@link SwingUtilities#invokeLater(Runnable)}. We recommend to avoid
	 * synchronize as this will can block indefinitely (see notice 1) which
	 * might lead to deadlocks.
	 * 
	 * 
	 * @param data
	 *            additional data, typically the new value associated with the
	 *            event
	 */
	public void notifyChange(T data);
}
