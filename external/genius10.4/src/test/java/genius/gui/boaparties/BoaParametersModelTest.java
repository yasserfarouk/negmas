package genius.gui.boaparties;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import genius.core.boaframework.BOAparameter;
import genius.core.repository.boa.ParameterList;

public class BoaParametersModelTest {
	/**
	 * Check that if we provide empty parameters, we get 1 resulting empty
	 * parameterlist.
	 */
	@Test
	public void test() {
		List<BOAparameter> params = new ArrayList<>();
		List<ParameterList> res = BoaParametersModel.getSettings(params);
		assertEquals(1, res.size());
		assertEquals(new ParameterList(), res.get(0));
	}

	/**
	 * Check that if we provide 1 parameter, we get all possible values.
	 */
	@Test
	public void test1Value() {
		List<BOAparameter> params = new ArrayList<>();
		params.add(new BOAparameter("v1", 0., 3., 1., "bla"));
		List<ParameterList> res = BoaParametersModel.getSettings(params);
		assertEquals(4, res.size());

		for (int n = 0; n < 4; n++) {
			assertEquals(1, res.get(n).size());
			assertEquals((Double) (double) n, res.get(n).get(0).getValue());
		}
	}

	/**
	 * Check that if we provide 2 parameters, we get all possible permutations.
	 */
	@Test
	public void test2Value() {
		List<BOAparameter> params = new ArrayList<>();
		params.add(new BOAparameter("v1", 0., 2., 1., "bla")); // 3
		params.add(new BOAparameter("v2", 10., 11., 1., "bla")); // 2
		List<ParameterList> res = BoaParametersModel.getSettings(params);
		assertEquals(6, res.size());

		for (int v1 = 0; v1 < 3; v1++) {
			for (int v2 = 0; v2 < 2; v2++) {
				int n = v1 * 2 + v2;
				assertEquals(2, res.get(n).size());
				System.out.println(res.get(n));
			}
		}
	}

}
