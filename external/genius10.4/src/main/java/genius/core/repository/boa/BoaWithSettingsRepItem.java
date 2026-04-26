package genius.core.repository.boa;

import java.util.Set;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.Global;
import genius.core.boaframework.BOA;
import genius.core.boaframework.BOAparameter;
import genius.core.boaframework.BoaType;
import genius.core.exceptions.InstantiateException;
import genius.core.repository.RepositoryFactory;

/**
 * A abstract rep item for a {@link BoaRepItem} including the parameter settings
 * to instantiate it. immutable.
 */
@XmlRootElement(name = "boawithsettings")
public class BoaWithSettingsRepItem<T extends BOA> {

	@XmlElement
	private ParameterList parameters = new ParameterList();

	@XmlElement
	private BoaRepItem<T> item;

	/**
	 * @return the {@link BoaRepItem}. Subclasses contain the proper type of
	 *         BoaRepItem.
	 */
	public BoaRepItem<T> getBoa() {
		return item;
	}

	// for serialization
	@SuppressWarnings("unused")
	protected BoaWithSettingsRepItem() {
	}

	/**
	 * Create just any with given type. Parameters are set to default for the
	 * given component.
	 * 
	 * @param type
	 *            must match T.
	 * @throws InstantiateException
	 */
	@SuppressWarnings("unchecked")
	public BoaWithSettingsRepItem(BoaType type) throws InstantiateException {
		item = (BoaRepItem<T>) RepositoryFactory.getBoaRepository().getBoaComponents(type).get(0);
		Set<BOAparameter> params;
		params = item.getInstance().getParameterSpec();
		for (BOAparameter p : params) {
			parameters.add(new ParameterRepItem(p.getName(), p.getLow()));
		}

	}

	public BoaWithSettingsRepItem(BoaRepItem<T> item, ParameterList parameters) {
		if (parameters == null)
			throw new NullPointerException("parameters=null");
		this.parameters = parameters;
		this.item = item;
	}

	/**
	 * @return parameter settings needed to instantiate this BoaRepItem.
	 */
	public ParameterList getParameters() {
		return parameters;
	}

	@Override
	public String toString() {
		return "BOA[" + getBoa() + "," + parameters + "]";
	}

	/**
	 * 
	 * @return a unique name for a BOA component. Both the referred class
	 *         component and the parameters make it unique.
	 * 
	 */
	public String getUniqueName() {
		String name = Global.nameOfClass(getBoa().getClassPath());
		for (ParameterRepItem p : parameters) {
			name = name + "_" + p.getName();
		}
		return name;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((parameters == null) ? 0 : parameters.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		BoaWithSettingsRepItem other = (BoaWithSettingsRepItem) obj;
		if (parameters == null) {
			if (other.parameters != null)
				return false;
		} else if (!parameters.equals(other.parameters))
			return false;
		return true;
	}

}
