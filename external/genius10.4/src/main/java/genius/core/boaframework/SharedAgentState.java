package genius.core.boaframework;

/**
 * When decoupling existing agents into their separate components, it often happens
 * that a component loosely depends on another component; for example an acceptance condition
 * can depend on a target utility calculated by the offering strategy.
 * 
 * To avoid code duplication a Shared Agent State class can be introduced containing the shared
 * code. In this case one of the components calculates the required data, while the other simply
 * requests the stored result.
 * 
 * Note that the only requirement by this class is the implementation of the name. The name
 * should be used to verify that the component which calculates (a part of) the results is
 * available.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx
 */
public abstract class SharedAgentState {
	
	protected String NAME;
	
	/**
	 * @return name of the SAS component.
	 */
	public String getName() {
		return NAME;
	}
}