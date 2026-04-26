package genius.core.boaframework;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import java.io.Serializable;

import genius.core.NegotiationResult;
import genius.core.repository.boa.BoaRepItem;

/**
 * Abstract superclass for BOA components. This should have been called
 * BOAcomponent but that class is already in use. Usually instances are created
 * using {@link BoaRepItem#getInstance()} and then calling
 * {{@link #init(NegotiationSession)}}
 **/
public abstract class BOA {
	/**
	 * Reference to the object which holds all information about the negotiation
	 */

	protected NegotiationSession negotiationSession;
	private Map<String, Double> parameters;

	/**
	 * Initializes the model. The method must be called once, and only once,
	 * immediately after creating an opponent model.
	 * 
	 * @param negotiationSession
	 *            reference to the state of the negotiation
	 * @param parameters
	 *            the init parameters used to configure this component
	 */
	protected void init(NegotiationSession negotiationSession,
			Map<String, Double> parameters) {
		this.negotiationSession = negotiationSession;
		this.parameters = parameters;
	}

	/**
	 * @return * The actual parameters that are used by this instance, as set by
	 *         call to {@link #init(NegotiationSession, Map)}.
	 * 
	 */
	public Map<String, Double> getParameters() {
		return parameters;
	}

	/**
	 * @return the parameter specifications of this BOA component. Default
	 *         implementation returns empty set. If a BOA component has
	 *         parameters, it should override this. This can be different from
	 *         the actual parameters used at runtime, which is passed through
	 *         {@link #init(NegotiationSession)} calls to the components.
	 * 
	 */
	public Set<BOAparameter> getParameterSpec() {
		return new HashSet<BOAparameter>();
	}

	/**
	 * Method called at the end of the negotiation. Ideal location to call the
	 * storeData method to receiveMessage the data to be saved.
	 * 
	 * @param result
	 *            of the negotiation.
	 */
	public void endSession(NegotiationResult result) {
	}

	/**
	 * Method used to store data that should be accessible in the next
	 * negotiation session on the same scenario. This method can be called
	 * during the negotiation, but it makes more sense to call it in the
	 * endSession method.
	 * 
	 * @param object
	 *            to be saved by this component.
	 */
	abstract public void storeData(Serializable object);

	/**
	 * Method used to load the saved object, possibly created in a previous
	 * negotiation session. The method returns null when such an object does not
	 * exist yet.
	 * 
	 * @return saved object or null when not available.
	 */
	abstract public Serializable loadData();

	/**
	 * 
	 * @return a short name for this component.
	 */
	abstract public String getName();

}