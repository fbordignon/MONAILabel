# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import logging
import os

import qt
import slicer
import vtk
from MONAITransformsLib import MonaiUtils
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


class MONAITransforms(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)

        self.parent.title = "MONAI Transforms"
        self.parent.categories = ["MONAI"]
        self.parent.dependencies = []
        self.parent.contributors = ["NVIDIA"]
        self.parent.helpText = """
This extension helps to run chain of MONAI transforms and visualize every stage over an image/label.
See more information in <a href="https://github.com/Project-MONAI/MONAILabel">module documentation</a>.
"""
        self.parent.acknowledgementText = """
Developed by NVIDIA
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # MONAITransforms1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MONAITransforms",
        sampleName="MONAITransforms1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "MONAITransforms1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="MONAITransforms1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="MONAITransforms1",
    )

    # MONAITransforms2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="MONAITransforms",
        sampleName="MONAITransforms2",
        thumbnailFileName=os.path.join(iconsPath, "MONAITransforms2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="MONAITransforms2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="MONAITransforms2",
    )


class MONAITransformsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.transforms = None
        self.tmpdir = slicer.util.tempDirectory("slicer-monai-transforms")

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MONAITransforms.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = MONAITransformsLogic()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.addTransformButton.connect("clicked(bool)", self.onAddTransform)
        self.ui.removeTransformButton.connect("clicked(bool)", self.onRemoveTransform)
        self.ui.moveUpButton.connect("clicked(bool)", self.onMoveUpTransform)
        self.ui.moveDownButton.connect("clicked(bool)", self.onMoveDownTransform)
        self.ui.modulesComboBox.connect("currentIndexChanged(int)", self.onSelectModule)
        self.ui.transformTable.connect("cellClicked(int, int)", self.onSelectTransform)
        self.ui.transformTable.connect("cellDoubleClicked(int, int)", self.onEditTransform)
        self.ui.importBundleButton.connect("clicked(bool)", self.onImportBundle)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        self.ui.importBundleButton.setIcon(self.icon("download.png"))
        self.ui.addTransformButton.setIcon(self.icon("icons8-insert-row-48.png"))
        self.ui.removeTransformButton.setIcon(self.icon("icons8-delete-row-48.png"))
        self.ui.editTransformButton.setIcon(self.icon("icons8-edit-row-48.png"))

        headers = ["Target", "Init Keys"]
        self.ui.transformTable.setColumnCount(len(headers))
        self.ui.transformTable.setHorizontalHeaderLabels(headers)
        self.ui.transformTable.setColumnWidth(0, 200)
        self.ui.transformTable.setEditTriggers(qt.QTableWidget.NoEditTriggers)
        self.ui.transformTable.setSelectionBehavior(qt.QTableView.SelectRows)

        self.refreshVersion()

    def cleanup(self):
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        self.ui.applyButton.enabled = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))

        self._parameterNode.EndModify(wasModified)

    def icon(self, name="MONAILabel.png"):
        # It should not be necessary to modify this method
        iconPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", name)
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def onApplyButton(self):
        pass

    def refreshVersion(self):
        print("Refreshing Version...")
        self.ui.monaiVersionComboBox.clear()
        version = MonaiUtils.version()
        self.ui.monaiVersionComboBox.addItem(version)
        self.ui.monaiVersionComboBox.setCurrentText(version)

        self.refreshTransforms()

        # bundle names
        bundles = MonaiUtils.list_bundles()
        self.ui.bundlesComboBox.clear()
        self.ui.bundlesComboBox.addItems(list(sorted({b[0] for b in bundles})))
        idx = max(0, self.ui.bundlesComboBox.findText("spleen_ct_segmentation"))
        self.ui.bundlesComboBox.setCurrentIndex(idx)

        self.ui.bundleStageComboBox.clear()
        self.ui.bundleStageComboBox.addItems(["pre", "post"])

    def refreshTransforms(self):
        if not self.ui.monaiVersionComboBox.currentText:
            return

        print("Refreshing Transforms...")
        self.transforms = MonaiUtils.list_transforms()

        self.ui.modulesComboBox.clear()
        self.ui.modulesComboBox.addItems(sorted(list({v["module"] for v in self.transforms.values()})))

        idx = max(0, self.ui.modulesComboBox.findText("monai.transforms.io.dictionary"))
        self.ui.modulesComboBox.setCurrentIndex(idx)
        # self.onSelectModule(self.ui.modulesComboBox.currentText)

    def onImportBundle(self):
        if not self.ui.monaiVersionComboBox.currentText:
            return
        name = self.ui.bundlesComboBox.currentText
        bundle_dir = os.path.join(self.tmpdir, "bundle")
        this_bundle = os.path.join(bundle_dir, name)
        if not os.path.exists(this_bundle):
            print(f"Downloading {name} to {bundle_dir}")
            MonaiUtils.download_bundle(name, bundle_dir)

        transforms = MonaiUtils.transforms_from_bundle(name, bundle_dir)

        table = self.ui.transformTable
        table.clearContents()
        table.setRowCount(len(transforms))

        pos = 0
        for t in transforms:
            name = t["_target_"]
            args = copy.copy(t)
            args.pop("_target_")

            print(f"Importing Transform: {name} => {args}")
            table.setItem(pos, 0, qt.QTableWidgetItem(name))
            table.setItem(pos, 1, qt.QTableWidgetItem(json.dumps(args)))
            pos += 1

    def onSelectModule(self):
        module = self.ui.modulesComboBox.currentText
        print(f"Selected Module: {module}")

        filtered = [k for k, v in self.transforms.items() if v["module"] == module]
        filtered = [f.replace(f"{module}.", "") for f in filtered]
        self.ui.transformsComboBox.clear()
        self.ui.transformsComboBox.addItems(filtered)

    def onSelectTransform(self, row, col):
        selected = True if row >= 0 and self.ui.transformTable.rowCount else False
        self.ui.editTransformButton.setEnabled(selected)
        self.ui.removeTransformButton.setEnabled(selected)
        self.ui.moveUpButton.setEnabled(selected and row > 0)
        self.ui.moveDownButton.setEnabled(selected and row < self.ui.transformTable.rowCount - 1)

    def onEditTransform(self, row, col):
        print(f"Selected Transform for Edit: {row}")

    def onAddTransform(self):
        print(f"Adding Transform: {self.ui.modulesComboBox.currentText}.{self.ui.transformsComboBox.currentText}")
        if not self.ui.modulesComboBox.currentText or not self.ui.transformsComboBox.currentText:
            return

        t = self.ui.transformsComboBox.currentText
        m = self.ui.modulesComboBox.currentText

        table = self.ui.transformTable
        pos = table.rowCount if table.currentRow() < 0 else table.currentRow()
        table.insertRow(pos)

        table.setItem(pos, 0, qt.QTableWidgetItem(f"{m}.{t}"))

    def onRemoveTransform(self):
        row = self.ui.transformTable.currentRow()
        if row < 0:
            return
        self.ui.transformTable.removeRow(row)
        self.onSelectTransform(-1, -1)

    def onMoveUpTransform(self):
        pass

    def onMoveDownTransform(self):
        pass

    def onApply(self):
        image = "/localhome/sachi/Datasets/Radiology/Task09_Spleen/imagesTr/spleen_2.nii.gz"
        label = "/localhome/sachi/Datasets/Radiology/Task09_Spleen/labelsTr/spleen_2.nii.gz"


class MONAITransformsLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        # if not parameterNode.GetParameter("Threshold"):
        #     parameterNode.SetParameter("Threshold", "100.0")
        pass

    def process(self):
        import time

        startTime = time.time()
        logging.info("Processing started")

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")


class MONAITransformsTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_MONAITransforms1()

    def test_MONAITransforms1(self):
        self.delayDisplay("Starting the test")
        self.delayDisplay("Test passed")
