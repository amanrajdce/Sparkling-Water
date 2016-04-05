/**
 * Created by root on 5/4/16.
 * This class implements the grid search for a deep learning model
 * managed by aman raj
 *
 */
package Utils;
import hex.Model;
import hex.grid.Grid;
import hex.deeplearning.DeepLearningModel.DeepLearningParameters;
import hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation;
import water.Job;
import water.fvec.H2OFrame;
import static java.lang.System.out;
import java.util.Arrays;
import java.util.HashMap;
import hex.grid.GridSearch;
import water.util.ArrayUtils;

public final class RunGridModelling {

    public static Model[] modelgrid(final H2OFrame train_data, final H2OFrame test_data, final String response )

    {   // parameters for tuning
        HashMap<String, Object[]> hyperParms = new HashMap<String, Object[]>() {{
            put("_hidden", new Integer[]{100,100,100});
            put("_l1", new Double[]{1e-6, 1e-5, 1e-4});
        }};

        // Name of used hyper parameters
        String[] hyperParamNames = hyperParms.keySet().toArray(new String[hyperParms.size()]);
        Arrays.sort(hyperParamNames);
        int hyperSpaceSize = ArrayUtils.crossProductSize(hyperParms);

        // defining common parameter for grid search
        DeepLearningParameters dlParams = new DeepLearningParameters();
        dlParams._train = train_data._key;
        dlParams._valid = test_data._key;
        dlParams._response_column = response;
        dlParams._epochs = 2;
        dlParams._balance_classes = true;
        dlParams._activation = Activation.RectifierWithDropout;
        // starting grid search here
        Job<Grid> gs = GridSearch.startGridSearch(null, dlParams, hyperParms);
        Grid<DeepLearningParameters> grid  = (Grid<DeepLearningParameters>) gs.get();

        // Checking if number of produced models match size of specified hyper space
        if (hyperSpaceSize == (grid.getModelCount() + grid.getFailureCount()))
            out.println("all the combinations used..");
        else
            out.println("all cominations were not used, some failed..");

        // Get built models
        Model[] models = grid.getModels();
        return models;
    }
}
