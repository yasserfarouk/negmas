package negotiator.boaframework.offeringstrategy.anac2013.inoxAgent;

import java.io.Serializable;

/**
 * Simple help class to facilitate saving and loading.
 * 
 * @author Ruben van Zessen, Mariana Branco
 *
 */
public class SaveHelper implements Serializable{

	private static final long serialVersionUID = -2510382202061071011L;
	/** Result of negotiations */
	private double result;
	/** Number of previous negotiations */
	private int number;
	
	/**
	 * Constructor, sets the result and number of previous negotiations.
	 */
	public SaveHelper(double res, int num) {
		result = res;
		number = num;
	}
	
	/**
	 * Function that returns the stored result.
	 */
	public double getResult() {
		return result;
	}
	
	/**
	 * Function that returns the stored number of previous negotiations.
	 */
	public int getNumber() {
		return number;
	}
}
