/*
 * NegotiatorException.java
 *
 * Created on November 17, 2006, 3:59 PM
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package genius.core.exceptions;

/**
 * reports problem with negotiator agent.
 * 
 * @author dmytro This is a generic class of nogotiation errors.
 */
public class NegotiatorException extends Exception {

	private static final long serialVersionUID = 5934438120399990013L;

	/** Creates a new instance of NegotiatorException */
	// Wouter: I think we dont need a constructor,
	// the constructor of Exception is good enough.
	public NegotiatorException(String message) {
		super(message);
	}

	public NegotiatorException(String message, Throwable e) {
		super(message, e);
	}
}